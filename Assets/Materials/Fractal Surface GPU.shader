Shader "Fractal/Fractal Surface GPU" {

	SubShader {
		CGPROGRAM
		#pragma surface ConfigureSurface Standard fullforwardshadows addshadow //表面 标准光照模型 前向渲染 加入阴影
		#pragma instancing_options assumeuniformscaling procedural:ConfigureProcedural//启用实例化 假设均匀缩放(避免额外计算) 自己的ConfigureProcedural函数 
		#pragma editor_sync_compilation//这个指令确保在编辑器中修改 Shader 时，编译是同步进行的。这可以避免在编辑 Shader 时出现异步编译导致的一些问题。

		#pragma target 4.5
		
		#include "FractalGPU.hlsl"

		struct Input {
			float3 worldPos;
		};

		float _Smoothness;

		//表面渲染所需要用到的函数
		void ConfigureSurface (Input input, inout SurfaceOutputStandard surface) {
			surface.Albedo = GetFractalColor().rgb;
			surface.Smoothness = surface.Smoothness = GetFractalColor().a;
		}
		ENDCG
	}

	FallBack "Diffuse"
}