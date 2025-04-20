using Unity.Burst;           // 引入Burst编译器，用于优化性能
using Unity.Collections;     // 引入用于管理原生内存的集合类型
using Unity.Jobs;            // 引入Job系统，用于多线程
using Unity.Mathematics;     // 引入数学库，提供高性能数学运算
using UnityEngine;

// 使用静态导入，简化数学函数调用
using static Unity.Mathematics.math;
using quaternion = Unity.Mathematics.quaternion;  // 使用Unity.Mathematics的四元数类型
using Random = UnityEngine.Random;                // 使用UnityEngine的随机数生成器

public class Fractal : MonoBehaviour {

    // 分形级别更新Job，使用Burst编译优化性能
    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast, CompileSynchronously = true)]
    struct UpdateFractalLevelJob : IJobFor {

        public float scale;           // 当前级别的缩放比例
        public float deltaTime;       // 当前帧时间增量

        [ReadOnly]
        public NativeArray<FractalPart> parents;  // 父球是第几层

        public NativeArray<FractalPart> parts;    // 自己所处第几层

        [WriteOnly]
        public NativeArray<float3x4> matrices;    // 用于给GPU实例化渲染的矩阵数组，只写

        public void Execute (int i) {
            FractalPart parent = parents[i / 5];//看一下自己的父球在父层是第几个球
            FractalPart part = parts[i];//看下自己在自己层第几个球
            
            // 算出要更新的自旋角度
            part.spinAngle += part.spinVelocity * deltaTime;

            float3 upAxis = mul(mul(parent.worldRotation, part.rotation), up());//父球局部Y轴和世界Y轴的差
            float3 sagAxis = cross(up(), upAxis);//下垂向量:父球局部Y轴与世界Y轴向量的叉积，表示节点应该向哪个方向下垂
            
            float sagMagnitude = length(sagAxis);//算出下垂向量的长度
            quaternion baseRotation;
            
            //如果 sagMag>0，生成绕 sagAxis 的小角度旋转 sagRot，并乘以父节点的世界旋转；否则直接用父旋转
            if (sagMagnitude > 0f) {
                sagAxis /= sagMagnitude;  // 归一化
                // 基于下垂轴和最大下垂角度创建旋转
                //这一行在做的是“创建一个绕某根轴的旋转”
                // sagAxis：表示分支应该向哪个方向“下垂”的轴（它是全局 up 向量和父分支局部“上”向量的叉积、再归一化得到的方向）。
                // part.maxSagAngle：该分支允许的最小和最大下垂角度范围（在 Inspector 中配置的弧度值）。
                // sagMagnitude：分支当前偏离垂直的程度（0~1 之间），越偏离垂直，下垂越明显。
                // 把它们相乘 part.maxSagAngle * sagMagnitude 得到本帧实际的下垂角度，
                // 然后传给quaternion.AxisAngle(axis, angle)就会返回一个四元数，表示绕 axis 旋转 angle 弧度的变换。
                // 这个 sagRotation 再与父节点的 worldRotation 相乘，就可以让子分支既继承父旋转又带有向下弯曲（sag）的效果。
                quaternion sagRotation =
                    quaternion.AxisAngle(sagAxis, part.maxSagAngle * sagMagnitude);
                baseRotation = mul(sagRotation, parent.worldRotation);//sagRotation就是重力偏移
            }
            else {
                // 下垂向量长度为0说明没有下垂，使用父旋转
                baseRotation = parent.worldRotation;
            }

            //先本地旋转＋自旋,再把上一步的结果，乘上父节点的基础旋转（包括继承的世界旋转＋下垂偏移），得出最终 worldRotation
            part.worldRotation = mul(baseRotation,
                mul(part.rotation, quaternion.RotateY(part.spinAngle))//本地旋转＋自旋
            );
            
            // 7. 计算世界位置 = 父级位置 + 本地偏移(沿 local Y 轴 * 1.5 * scale)
            part.worldPosition =
                parent.worldPosition +
                mul(part.worldRotation, float3(0f, 1.5f * scale, 0f));
                
            parts[i] = part;

            float3x3 r = float3x3(part.worldRotation) * scale;
            matrices[i] = float3x4(r.c0, r.c1, r.c2, part.worldPosition);
        }
    }

    // 分形部分结构，存储位置和旋转信息
    struct FractalPart {
        public float3 worldPosition;              // 世界坐标位置
        public quaternion rotation, worldRotation; // 局部旋转和世界旋转
        public float maxSagAngle, spinAngle, spinVelocity;  // 最大下垂角度、当前自旋角度和自旋速度
    }

    // 着色器属性ID缓存 (这样做的好处是避免了每次使用属性名称时的字符串查找，提高了性能。)
    static readonly int
        colorAId = Shader.PropertyToID("_ColorA"),
        colorBId = Shader.PropertyToID("_ColorB"),
        matricesId = Shader.PropertyToID("_Matrices"),
        sequenceNumbersId = Shader.PropertyToID("_SequenceNumbers");

    // 存储的5层的旋转集合
    static quaternion[] rotations = {
        quaternion.identity,                          // 中央分支
        quaternion.RotateZ(-0.5f * PI), quaternion.RotateZ(0.5f * PI),  // Z轴正负方向
        quaternion.RotateX(0.5f * PI), quaternion.RotateX(-0.5f * PI)   // X轴正负方向
    };

    static MaterialPropertyBlock propertyBlock;  // 材质属性块，用于实例化绘制

    // 分形深度配置
    [SerializeField, Range(3, 8)]
    int depth = 4;

    // 网格配置
    [SerializeField]
    Mesh mesh, leafMesh;  // 常规网格和叶子网格

    // 材质配置
    [SerializeField]
    Material material;

    // 颜色渐变配置
    [SerializeField]
    Gradient gradientA, gradientB;

    // 叶子颜色配置
    [SerializeField]
    Color leafColorA, leafColorB;

    // 下垂角度配置
    [SerializeField, Range(0f, 90f)]
    float maxSagAngleA = 15f, maxSagAngleB = 25f;

    // 自旋速度配置
    [SerializeField, Range(0f, 90f)]
    float spinSpeedA = 20f, spinSpeedB = 25f;

    // 反向自旋概率
    [SerializeField, Range(0f, 1f)]
    float reverseSpinChance = 0.25f;

    //主要被CPU所用的数据,parts用于计算和更新, 双重数组存储4层所有小球的数据 [[FractalPart],[5个FractalPart],[25个FractalPart],[125个FractalPart]] ,二维数组里的第一维数组代表5个层,第二维数组代表存着这一层存的小球数量
    NativeArray<FractalPart>[] parts;//要注意!!!这其实是个双重数组,后面的[]是个数组,里面包着5个NativeArray<FractalPart>的一维数组,第二数组里面包的才是FractalPart单个小球数据

    // 多级变换矩阵数组
    NativeArray<float3x4>[] matrices;//而matrices是为了渲染优化而存在的,主要为GPU所用的数据,存储GPU所需要的矩形数据,包含了位置、旋转和缩放信息,作为中间存储，连接了CPU端的分形部分信息和GPU端的渲染数据。

    // 计算着色器缓冲区
    ComputeBuffer[] matricesBuffers;

//     据流转关系
// 整个数据流转过程是：

// parts (CPU内存，计算用) →
// matrices (CPU内存，格式转换) →
// matricesBuffers (GPU内存，渲染用)
// 为什么需要三个不同的数据结构？
// parts：结构化数据，适合CPU计算和逻辑处理
// matrices：格式化的矩阵，是CPU和GPU的中间形式
// matricesBuffers：GPU可直接访问的内存，专为高效渲染设计据流转关系
// 整个数据流转过程是：

// parts (CPU内存，计算用) →
// matrices (CPU内存，格式转换) →
// matricesBuffers (GPU内存，渲染用)
// 为什么需要三个不同的数据结构？
// parts：结构化数据，适合CPU计算和逻辑处理
// matrices：格式化的矩阵，是CPU和GPU的中间形式
// matricesBuffers：GPU可直接访问的内存，专为高效渲染设计

    // 序列号数组，用于在着色器中创建变化
    Vector4[] sequenceNumbers;

    // 启用时初始化
    void OnEnable () {
        // 创建数组
        parts = new NativeArray<FractalPart>[depth]; //先给外围数组设置五层
        matrices = new NativeArray<float3x4>[depth];
        matricesBuffers = new ComputeBuffer[depth];
        sequenceNumbers = new Vector4[depth];
        
        int stride = 12 * 4;  // 每个矩阵的字节大小
        
        // 再根据每一层小球的数量设置内围数组里存几个小球数据
        // length指的是你这一层有几个小球,一层有1个 二层有5个 三层有25个,
        for (int i = 0, length = 1; i < parts.Length; i++, length *= 5) {
            // 每层的球数是上一级的5倍
            parts[i] = new NativeArray<FractalPart>(length, Allocator.Persistent);
            matrices[i] = new NativeArray<float3x4>(length, Allocator.Persistent);
            matricesBuffers[i] = new ComputeBuffer(length, stride);
            // 为每一层生成随机序列号
            sequenceNumbers[i] =
                new Vector4(Random.value, Random.value, Random.value, Random.value);
        }

        // 创建根部分圆体,创建层级的同时也会设置旋转速度和角度
        parts[0][0] = CreatePart(0);
        
        // 创建所有子部分
        for (int li = 1; li < parts.Length; li++) {
            NativeArray<FractalPart> levelParts = parts[li];//每层所有的球数据
            for (int fpi = 0; fpi < levelParts.Length; fpi += 5) {//每5个球隶属于一个父球,
                for (int ci = 0; ci < 5; ci++) {//五个球中每个球
                    levelParts[fpi + ci] = CreatePart(ci);//看来这个CreatePart接受的参数是周围的五个小球,你给他传上面那个小球那他就告诉你他会按照y轴转,转多少度转多快
                }
            }
        }

        // 初始化属性块（如果需要）
        propertyBlock ??= new MaterialPropertyBlock();
    }

    // 禁用时释放资源
    void OnDisable () {
        for (int i = 0; i < matricesBuffers.Length; i++) {
            matricesBuffers[i].Release();
            parts[i].Dispose();
            matrices[i].Dispose();
        }
        parts = null;
        matrices = null;
        matricesBuffers = null;
        sequenceNumbers = null;
    }

    // 验证更改时重新初始化
    void OnValidate () {
        if (parts != null && enabled) {
            OnDisable();
            OnEnable();
        }
    }

    // 这个CreatePart接受的参数是周围的五个小球,你给他传上面那个小球那他就告诉你他会按照y轴转,转多少度转多快什么位置
    FractalPart CreatePart (int childIndex) => new FractalPart {
        // 在配置范围内随机下垂角度
        maxSagAngle = radians(Random.Range(maxSagAngleA, maxSagAngleB)),
        // 拿到第几层的旋转
        rotation = rotations[childIndex],
        // 随机自旋速度，有一定概率反向旋转
        spinVelocity =
            (Random.value < reverseSpinChance ? -1f : 1f) *
            radians(Random.Range(spinSpeedA, spinSpeedB))
    };

    // 每帧更新
    void Update () {
        float deltaTime = Time.deltaTime;
        
        // 更新根部分
        FractalPart rootPart = parts[0][0];
        rootPart.spinAngle += rootPart.spinVelocity * deltaTime;//根部开始旋转（此时大球面向角度=旋转力度*时间走动）
        // quaternion.RotateY(rootPart.spinAngle)
        // 生成一个绕 Y 轴旋转 spinAngle 弧度的新四元数（表示部件在本地自身坐标系上的旋转）。
        // · mul(rootPart.rotation, spinQuat)
        // 先进行自旋，再应用部件的“基准”朝向 rootPart.rotation。
        // · mul(transform.rotation, localQuat)
        // 最后再把整个部件的本地朝向，乘上挂载此脚本的 GameObject 的世界旋转 transform.rotation，这样就变成了部件在世界空间里的旋转 worldRotation。
        // 注意 mul(q1, q2) 在 Unity.Mathematics 里就是四元数乘法，等价于 q1 * q2：先绕 q2 旋，再绕 q1 旋。
        rootPart.worldRotation = mul(transform.rotation,
            mul(rootPart.rotation, quaternion.RotateY(rootPart.spinAngle))
        );
        rootPart.worldPosition = transform.position;//将最大球的位置重新赋值在原始根GameObj上
        parts[0][0] = rootPart;
        
        // 3. 把根节点的变换信息转换成矩阵，存入 matrices[0][0]
        float objectScale = transform.lossyScale.x;                       // 全局缩放
        float3x3 r = float3x3(rootPart.worldRotation) * objectScale;     // 3x3 旋转+缩放矩阵
        matrices[0][0] = float3x4(r.c0, r.c1, r.c2, rootPart.worldPosition);// 3x4 矩阵（包含平移）

        // 使用Job系统更新所有子级
        float scale = objectScale;
        JobHandle jobHandle = default;
        for (int li = 1; li < parts.Length; li++) {//五层每层1个jobHandle
            scale *= 0.5f;  // 每级缩小一半
            // 创建每层的jobHandle,指出自己的父层和自己所在的层,也将矩形数据准备好
            jobHandle = new UpdateFractalLevelJob {
                deltaTime = deltaTime,
                scale = scale,
                parents = parts[li - 1],
                parts = parts[li],
                matrices = matrices[li]
            }.ScheduleParallel(parts[li].Length, 5, jobHandle);//5次渲染
        }
        // 等待所有Job完成
        jobHandle.Complete();

        // 设置包围盒
        var bounds = new Bounds(rootPart.worldPosition, 3f * objectScale * Vector3.one);
        int leafIndex = matricesBuffers.Length - 1;
        
        // 绘制所有级别
        for (int i = 0; i < matricesBuffers.Length; i++) {
            ComputeBuffer buffer = matricesBuffers[i];
            // 更新矩阵缓冲区
            buffer.SetData(matrices[i]);
            
            Color colorA, colorB;
            Mesh instanceMesh;
            
            // 根据是否是叶子节点选择不同的外观
            if (i == leafIndex) {
                // 叶子节点
                colorA = leafColorA;
                colorB = leafColorB;
                instanceMesh = leafMesh;
            }
            else {
                // 普通节点，使用渐变颜色
                float gradientInterpolator = i / (matricesBuffers.Length - 2f);
                colorA = gradientA.Evaluate(gradientInterpolator);
                colorB = gradientB.Evaluate(gradientInterpolator);
                instanceMesh = mesh;
            }
            
            // 设置着色器属性
            propertyBlock.SetColor(colorAId, colorA);
            propertyBlock.SetColor(colorBId, colorB);
            propertyBlock.SetBuffer(matricesId, buffer);
            propertyBlock.SetVector(sequenceNumbersId, sequenceNumbers[i]);
            
            // 使用GPU实例化绘制
            Graphics.DrawMeshInstancedProcedural(
                instanceMesh, 0, material, bounds, buffer.count, propertyBlock
            );
        }
    }
}

/*
 * ==========================================================================
 * 代码分析:
 * ==========================================================================
 * 
 * Fractal组件概述:
 * ---------------
 * 该组件实现了一个分形树结构生成系统，具有有机的动态效果。它使用Unity的Job System和
 * Burst编译器进行性能优化，并通过GPU实例化技术高效渲染大量相似对象。
 *
 * 主要技术特点:
 * ------------
 * 1. 使用Job System和Burst编译实现多线程处理，显著提高性能
 * 2. 利用GPU实例化(DrawMeshInstancedProcedural)渲染大量对象
 * 3. 应用NativeArray进行内存管理，避免GC压力
 * 4. 采用Unity.Mathematics库提供高性能数学运算
 *
 * 分形结构设计:
 * ------------
 * - 层级结构: 从单一根部分开始，每个部分生成5个子部分，形成递归树结构
 * - 部分布局: 一个中心向上的分支，四个向不同方向的分支(±X和±Z轴)
 * - 缩放规律: 每个子级的缩放比例是父级的一半
 * - 特殊处理: 树的叶子节点(最深层级)使用不同的网格和颜色
 *
 * 有机动态效果:
 * ------------
 * 1. 自旋(Spin)效果:
 *    - 每个部分绕Y轴旋转
 *    - 随机的自旋速度和方向
 *    - 通过spinVelocity和spinAngle控制
 *
 * 2. 下垂(Sag)效果:
 *    - 模拟重力影响，使分支向下弯曲
 *    - 通过sagAxis和maxSagAngle控制
 *    - 根据上轴方向偏离垂直程度计算下垂量
 *
 * 3. 随机性:
 *    - 每个部分的下垂角度在配置范围内随机化
 *    - 自旋速度随机化
 *    - 部分自旋方向随机反转
 *    - 每个深度级别有随机的序列号，用于着色器中的变化
 *
 * 渲染技术:
 * --------
 * - 使用MaterialPropertyBlock避免材质实例化
 * - 向GPU传递矩阵数据进行高效实例化渲染
 * - 根据层级应用颜色渐变
 * - 支持特殊的叶子节点渲染
 *
 * 性能考量:
 * --------
 * 1. 计算密集型操作使用Job System和Burst编译优化
 * 2. 使用ReadOnly和WriteOnly属性标记优化内存访问
 * 3. 通过GPU实例化减少绘制调用(draw calls)
 * 4. 使用NativeArray避免托管内存分配和GC
 *
 * 适用场景:
 * --------
 * - 程序化生成植物、树木等自然元素
 * - 创建具有有机变化的艺术效果
 * - 分形可视化和教育演示
 * - 作为高级Unity渲染和多线程技术的示例
 */