

__device__  __forceinline__ void outer_product(
	float4* X_frag, 
	float4* G_frag, 
	float4 accu[][16]) 
{
	accu[0][0].x += X_frag[0].x*G_frag[0].x;
	accu[0][0].z += X_frag[0].y*G_frag[0].x;
	accu[0][0].y += X_frag[0].z*G_frag[0].x;
	accu[0][0].w += X_frag[0].w*G_frag[0].x;

	accu[0][1].x += X_frag[1].x*G_frag[0].x;
	accu[0][1].z += X_frag[1].y*G_frag[0].x;
	accu[0][1].y += X_frag[1].z*G_frag[0].x;
	accu[0][1].w += X_frag[1].w*G_frag[0].x;

	accu[0][2].x += X_frag[0].x*G_frag[0].y;
	accu[0][2].z += X_frag[0].y*G_frag[0].y;
	accu[0][2].y += X_frag[0].z*G_frag[0].y;
	accu[0][2].w += X_frag[0].w*G_frag[0].y;

	accu[0][3].x += X_frag[1].x*G_frag[0].y;
	accu[0][3].z += X_frag[1].y*G_frag[0].y;
	accu[0][3].y += X_frag[1].z*G_frag[0].y;
	accu[0][3].w += X_frag[1].w*G_frag[0].y;

	accu[0][4].x += X_frag[0].x*G_frag[0].z;
	accu[0][4].z += X_frag[0].y*G_frag[0].z;
	accu[0][4].y += X_frag[0].z*G_frag[0].z;
	accu[0][4].w += X_frag[0].w*G_frag[0].z;

	accu[0][5].x += X_frag[1].x*G_frag[0].z;
	accu[0][5].z += X_frag[1].y*G_frag[0].z;
	accu[0][5].y += X_frag[1].z*G_frag[0].z;
	accu[0][5].w += X_frag[1].w*G_frag[0].z;

	accu[0][6].x += X_frag[0].x*G_frag[0].w;
	accu[0][6].z += X_frag[0].y*G_frag[0].w;
	accu[0][6].y += X_frag[0].z*G_frag[0].w;
	accu[0][6].w += X_frag[0].w*G_frag[0].w;

	accu[0][7].x += X_frag[1].x*G_frag[0].w;
	accu[0][7].z += X_frag[1].y*G_frag[0].w;
	accu[0][7].y += X_frag[1].z*G_frag[0].w;
	accu[0][7].w += X_frag[1].w*G_frag[0].w;

	//
	accu[0][8].x += X_frag[0].x*G_frag[1].x;
	accu[0][8].z += X_frag[0].y*G_frag[1].x;
	accu[0][8].y += X_frag[0].z*G_frag[1].x;
	accu[0][8].w += X_frag[0].w*G_frag[1].x;

	accu[0][9].x += X_frag[1].x*G_frag[1].x;
	accu[0][9].z += X_frag[1].y*G_frag[1].x;
	accu[0][9].y += X_frag[1].z*G_frag[1].x;
	accu[0][9].w += X_frag[1].w*G_frag[1].x;

	accu[0][10].x += X_frag[0].x*G_frag[1].y;
	accu[0][10].z += X_frag[0].y*G_frag[1].y;
	accu[0][10].y += X_frag[0].z*G_frag[1].y;
	accu[0][10].w += X_frag[0].w*G_frag[1].y;

	accu[0][11].x += X_frag[1].x*G_frag[1].y;
	accu[0][11].z += X_frag[1].y*G_frag[1].y;
	accu[0][11].y += X_frag[1].z*G_frag[1].y;
	accu[0][11].w += X_frag[1].w*G_frag[1].y;

	accu[0][12].x += X_frag[0].x*G_frag[1].z;
	accu[0][12].z += X_frag[0].y*G_frag[1].z;
	accu[0][12].y += X_frag[0].z*G_frag[1].z;
	accu[0][12].w += X_frag[0].w*G_frag[1].z;

	accu[0][13].x += X_frag[1].x*G_frag[1].z;
	accu[0][13].z += X_frag[1].y*G_frag[1].z;
	accu[0][13].y += X_frag[1].z*G_frag[1].z;
	accu[0][13].w += X_frag[1].w*G_frag[1].z;

	accu[0][14].x += X_frag[0].x*G_frag[1].w;
	accu[0][14].z += X_frag[0].y*G_frag[1].w;
	accu[0][14].y += X_frag[0].z*G_frag[1].w;
	accu[0][14].w += X_frag[0].w*G_frag[1].w;

	accu[0][15].x += X_frag[1].x*G_frag[1].w;
	accu[0][15].z += X_frag[1].y*G_frag[1].w;
	accu[0][15].y += X_frag[1].z*G_frag[1].w;
	accu[0][15].w += X_frag[1].w*G_frag[1].w;




	//////
	accu[1][0].x += X_frag[2].x*G_frag[2].x;
	accu[1][0].z += X_frag[2].y*G_frag[2].x;
	accu[1][0].y += X_frag[2].z*G_frag[2].x;
	accu[1][0].w += X_frag[2].w*G_frag[2].x;

	accu[1][1].x += X_frag[3].x*G_frag[2].x;
	accu[1][1].z += X_frag[3].y*G_frag[2].x;
	accu[1][1].y += X_frag[3].z*G_frag[2].x;
	accu[1][1].w += X_frag[3].w*G_frag[2].x;

	accu[1][2].x += X_frag[2].x*G_frag[2].y;
	accu[1][2].z += X_frag[2].y*G_frag[2].y;
	accu[1][2].y += X_frag[2].z*G_frag[2].y;
	accu[1][2].w += X_frag[2].w*G_frag[2].y;

	accu[1][3].x += X_frag[3].x*G_frag[2].y;
	accu[1][3].z += X_frag[3].y*G_frag[2].y;
	accu[1][3].y += X_frag[3].z*G_frag[2].y;
	accu[1][3].w += X_frag[3].w*G_frag[2].y;

	accu[1][4].x += X_frag[2].x*G_frag[2].z;
	accu[1][4].z += X_frag[2].y*G_frag[2].z;
	accu[1][4].y += X_frag[2].z*G_frag[2].z;
	accu[1][4].w += X_frag[2].w*G_frag[2].z;

	accu[1][5].x += X_frag[3].x*G_frag[2].z;
	accu[1][5].z += X_frag[3].y*G_frag[2].z;
	accu[1][5].y += X_frag[3].z*G_frag[2].z;
	accu[1][5].w += X_frag[3].w*G_frag[2].z;

	accu[1][6].x += X_frag[2].x*G_frag[2].w;
	accu[1][6].z += X_frag[2].y*G_frag[2].w;
	accu[1][6].y += X_frag[2].z*G_frag[2].w;
	accu[1][6].w += X_frag[2].w*G_frag[2].w;

	accu[1][7].x += X_frag[3].x*G_frag[2].w;
	accu[1][7].z += X_frag[3].y*G_frag[2].w;
	accu[1][7].y += X_frag[3].z*G_frag[2].w;
	accu[1][7].w += X_frag[3].w*G_frag[2].w;

	//
	accu[1][8].x += X_frag[2].x*G_frag[3].x;
	accu[1][8].z += X_frag[2].y*G_frag[3].x;
	accu[1][8].y += X_frag[2].z*G_frag[3].x;
	accu[1][8].w += X_frag[2].w*G_frag[3].x;

	accu[1][9].x += X_frag[3].x*G_frag[3].x;
	accu[1][9].z += X_frag[3].y*G_frag[3].x;
	accu[1][9].y += X_frag[3].z*G_frag[3].x;
	accu[1][9].w += X_frag[3].w*G_frag[3].x;

	accu[1][10].x += X_frag[2].x*G_frag[3].y;
	accu[1][10].z += X_frag[2].y*G_frag[3].y;
	accu[1][10].y += X_frag[2].z*G_frag[3].y;
	accu[1][10].w += X_frag[2].w*G_frag[3].y;

	accu[1][11].x += X_frag[3].x*G_frag[3].y;
	accu[1][11].z += X_frag[3].y*G_frag[3].y;
	accu[1][11].y += X_frag[3].z*G_frag[3].y;
	accu[1][11].w += X_frag[3].w*G_frag[3].y;

	accu[1][12].x += X_frag[2].x*G_frag[3].z;
	accu[1][12].z += X_frag[2].y*G_frag[3].z;
	accu[1][12].y += X_frag[2].z*G_frag[3].z;
	accu[1][12].w += X_frag[2].w*G_frag[3].z;

	accu[1][13].x += X_frag[3].x*G_frag[3].z;
	accu[1][13].z += X_frag[3].y*G_frag[3].z;
	accu[1][13].y += X_frag[3].z*G_frag[3].z;
	accu[1][13].w += X_frag[3].w*G_frag[3].z;

	accu[1][14].x += X_frag[2].x*G_frag[3].w;
	accu[1][14].z += X_frag[2].y*G_frag[3].w;
	accu[1][14].y += X_frag[2].z*G_frag[3].w;
	accu[1][14].w += X_frag[2].w*G_frag[3].w;

	accu[1][15].x += X_frag[3].x*G_frag[3].w;
	accu[1][15].z += X_frag[3].y*G_frag[3].w;
	accu[1][15].y += X_frag[3].z*G_frag[3].w;
	accu[1][15].w += X_frag[3].w*G_frag[3].w;
}