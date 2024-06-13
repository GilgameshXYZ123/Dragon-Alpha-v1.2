#define float4_elem_mul(a, b) float4{ a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w }

#define wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d) {\
	float4 m00 = float4_elem_mul(g0, d);\
	float4 m10 = float4_elem_mul(g1, d);\
	float4 m20 = float4_elem_mul(g2, d);\
	float4 m30 = float4_elem_mul(g3, d);\
	v0.x += (m00.x + m00.y + m00.z);\
	v0.y += (m10.x + m10.y + m10.z);\
	v0.z += (m20.x + m20.y + m20.z);\
	v0.w += (m30.x + m30.y + m30.z);\
	v1.x += (m00.y - m00.z - m00.w);\
	v1.y += (m10.y - m10.z - m10.w);\
	v1.z += (m20.y - m20.z - m20.w);\
	v1.w += (m30.y - m30.z - m30.w);}

#define WG4_GxW(v0, v1, g0, g1, g2, g3, d) {\
	float4 m00 = float4_elem_mul(g0, d);\
	float4 m10 = float4_elem_mul(g1, d);\
	float4 m20 = float4_elem_mul(g2, d);\
	float4 m30 = float4_elem_mul(g3, d);\
	v0.x += (m00.x + m00.y + m00.z);\
	v0.y += (m10.x + m10.y + m10.z);\
	v0.z += (m20.x + m20.y + m20.z);\
	v0.w += (m30.x + m30.y + m30.z);\
	v1.x += (m00.y - m00.z - m00.w);\
	v1.y += (m10.y - m10.z - m10.w);\
	v1.z += (m20.y - m20.z - m20.w);\
	v1.w += (m30.y - m30.z - m30.w);}

#define G_winograd4_W(v0, v1, g0, g1, g2, g3, d) {\
	float2 m00 = float2{ g0.y*d.y, g0.z*d.z };\
	float2 m10 = float2{ g1.y*d.y, g1.z*d.z };\
	float2 m20 = float2{ g2.y*d.y, g2.z*d.z };\
	float2 m30 = float2{ g3.y*d.y, g3.z*d.z };\
	v0.x += g0.x*d.x; v0.y += g1.x*d.x; v0.z += g2.x*d.x; v0.w += g3.x*d.x;\
	v1.x -= g0.w*d.w; v1.y -= g1.w*d.w; v1.z -= g2.w*d.w; v1.w -= g3.w*d.w;\
	v0.x += m00.x; v0.x += m00.y; v1.x += m00.x; v1.x -= m00.y;\
	v0.y += m10.x; v0.y += m10.y; v1.y += m10.x; v1.y -= m10.y;\
	v0.z += m20.x; v0.z += m20.y; v1.z += m20.x; v1.z -= m20.y;\
	v0.w += m30.x; v0.w += m30.y; v1.w += m30.x; v1.w -= m30.y; }


#define G_winograd4_W_V2(v0, v1, g0, g1, g2, g3, d) {\
	v0.x += g0.x*d.x; v0.y += g1.x*d.x; v0.z += g2.x*d.x; v0.w += g3.x*d.x;\
	v1.x -= g0.w*d.w; v1.y -= g1.w*d.w; v1.z -= g2.w*d.w; v1.w -= g3.w*d.w;\
	float4 t0 = float4{ g0.y*d.y, g1.y*d.y, g2.y*d.y, g3.y*d.y }; \
	float4 t1 = float4{ g0.z*d.z, g1.z*d.z, g2.z*d.z, g3.z*d.z }; \
	v0.x += t0.x; v0.y += t0.y; v0.z += t0.z; v0.w += t0.w;\
	v1.x += t0.x; v1.y += t0.y; v1.z += t0.z; v1.w += t0.w;\
	v0.x += t1.x; v0.y += t1.y; v0.z += t1.z; v0.w += t1.w; \
	v1.x -= t1.x; v1.y -= t1.y; v1.z -= t1.z; v1.w -= t1.w; }

//(1) v0: (g0 - g3).x * d.x		v1: (g0 - g3).w * d.w
//(2) t0: (g0 - g3).y * d.y		t1: (g0 - g3).z * d.z
//d.{x, y, z, w}: {x, w} for v; {y, z} for t
//g.{x, y, z, w}: {x, w} for v; {y, z} for t
#define G_winograd4_W_V3(v0, v1, t0, t1, g0, g1, g2, g3, d) {\
	v0.x += g0.x*d.x; v0.y += g1.x*d.x; v0.z += g2.x*d.x; v0.w += g3.x*d.x;\
	v1.x -= g0.w*d.w; v1.y -= g1.w*d.w; v1.z -= g2.w*d.w; v1.w -= g3.w*d.w;\
	t0.x += g0.y*d.y; t0.y += g1.y*d.y; t0.z += g2.y*d.y; t0.w += g3.y*d.y;\
	t1.x += g0.z*d.z; t1.y += g1.z*d.z; t1.z += g2.z*d.z; t1.w += g3.z*d.z; }