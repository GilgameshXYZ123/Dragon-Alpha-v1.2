/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl.math;

import z.util.lang.annotation.Passed;

/**
 * Y[N, OH, OW, OC] = conv3D(X[N, IH, IW, IC], W[OC, FH, FW, IC], [ph, pw, sh, sw]).
 * @author Gilgamesh
 */
public final class Cuda_conv3D {
    private Cuda_conv3D() {}
    
    //<editor-fold defaultstate="collapsed" desc="Common">
    public static final float psu(int IH, int IW, int OH, int OW, int FH, int FW, int sh, int sw) {//padding scale up
        int input_size = IH * IW;
        int pad_size = ((OH - 1)*sh + FH) * ((OW - 1)*sw + FW);
        return 1.0f * pad_size / input_size;
    }
    
    public static final float psu_s1(int IH, int IW, int OH, int OW, int FH, int FW) {//padding scale up
        int input_size = IH * IW;
        int pad_size = ((OH - 1) + FH) * ((OW - 1) + FW);
        return 1.0f * pad_size / input_size;
    }
    
    public static int[] img2col_matrix_dim(
            int OH, int OW, 
            int FH, int FW, 
            int N, int IC, int OC) {
        int GN = OC;
        int GM = N  * OH * OW;
        int GK = FH * FW * IC;
        return new int[]{ GN, GM, GK };
    }
    
    public static int[] output_feature_dim(
            int IH, int IW, 
            int FH, int FW, 
            int N, int OC,
            int sh, int sw, int ph, int pw) {
        int OH = (IH + (ph << 1) - FH) / sh + 1;
        int OW = (IW + (pw << 1) - FW) / sw + 1;
        return new int[]{ N, OH, OW, OC };
    }
    
    public static int[] input_feature_size(
            int OH, int OW, 
            int FH, int FW, 
            int N, int IC,
            int sh, int sw, int ph, int pw) {
        int IH = (OH - 1)*sh + FH - 2*ph;
        int IW = (OW - 1)*sw + FW - 2*pw;
        return new int[]{ N, IH, IW, IC };
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="computational intensity">
    public static final float double_buf_coef = 1.15f;
    public static final float out_product_16x8x8_coef = 0.9f;
    
    public static final float GEMM_intensity = 32 * double_buf_coef;//1024 / (4 + 4) / 4 = 32 opt/byte
    
    public static final float S4_intensity_f3x2 = 21.33333f * double_buf_coef;//1024 / (2*2 + 4*2) / 4 = 21.33333 opt/byte
    public static final float S4_intensity_f2x3 = 18.28571f * double_buf_coef;//1024 / (3*2 + 4*2) / 4 = 18.28571 opt/byte

    public static final float S4_intensity_f6x2 = 23.27273f * double_buf_coef;//1024 / (2*2 + 7) / 4 = 21.33333 opt/byte
    public static final float S4_intensity_f4x3 = 18.28571f * double_buf_coef;//1024 / (3*2 + 6) / 4 = 18.28571 opt/byte
    
    public static final float S8_intensity_f7x2 = 21.33333f * double_buf_coef;//1024 / (2*2 + 8) / 4 = 21.33333 opt/byte
    public static final float S8_intensity_f6x3 = 18.28571f * double_buf_coef;//1024 / (3*2 + 8) / 4 = 18.28571 opt/byte
    public static final float S8_intensity_f5x4 = 16.00000f * double_buf_coef;//1024 / (4*2 + 8) / 4 = 16.00000 opt/byte
    public static final float S8_intensity_f4x5 = 14.22222f * double_buf_coef;//1024 / (5*2 + 8) / 4 = 14.22222 opt/byte
    public static final float S8_intensity_f3x6 = 12.80000f * double_buf_coef;//1024 / (6*2 + 8) / 4 = 12.80000 opt/byte
    public static final float S8_intensity_f2x7 = 11.63636f * double_buf_coef;//1024 / (7*2 + 8) / 4 = 11.63636 opt/byte

    public static final float S8_intensity_f8x5 = 16.00000f * double_buf_coef;//2048 / (5*4 + 12) / 4 = 16.00000 opt/byte
    public static final float S8_intensity_f6x6 = 14.62857f * double_buf_coef;//2048 / (6*4 + 11) / 4 = 14.62857 opt/byte
    public static final float S8_intensity_f4x7 = 13.47368f * double_buf_coef;//2048 / (7*4 + 10) / 4 = 13.47368 opt/byte

    public static final float S16_intensity_f5xC =  9.14286f;//1024 / (12 + 16) / 4 =  9.14286 opt/byte
    public static final float S16_intensity_f6xB =  9.48148f;//1024 / (11 + 16) / 4 =  9.48148 opt/byte
    public static final float S16_intensity_f7xA =  9.84615f;//1024 / (10 + 16) / 4 =  9.84615 opt/byte
    public static final float S16_intensity_f8x9 = 10.24000f;//1024 / ( 9 + 16) / 4 = 10.24000 opt/byte
    public static final float S16_intensity_f9x8 = 10.66667f;//1024 / ( 8 + 16) / 4 = 10.66667 opt/byte
    public static final float S16_intensity_fAx7 = 11.13043f;//1024 / ( 7 + 16) / 4 = 11.13043 opt/byte

    public static final float S16_intensity_f5xC_c64 = 12.80000f * out_product_16x8x8_coef;//2048 / (12*2 + 16) / 4 = 12.80000 opt/byte
    public static final float S16_intensity_f6xB_c64 = 13.47368f * out_product_16x8x8_coef;//2048 / (11*2 + 16) / 4 = 13.47368 opt/byte
    public static final float S16_intensity_f7xA_c64 = 14.22222f * out_product_16x8x8_coef;//2048 / (10*2 + 16) / 4 = 14.22222 opt/byte
    public static final float S16_intensity_f8x9_c64 = 15.05882f * out_product_16x8x8_coef;//2048 / ( 9*2 + 16) / 4 = 15.05882 opt/byte
    public static final float S16_intensity_f9x8_c64 = 16.00000f * out_product_16x8x8_coef;//2048 / ( 8*2 + 16) / 4 = 16.00000 opt/byte
    public static final float S16_intensity_fAx7_c64 = 17.06667f * out_product_16x8x8_coef;//2048 / ( 7*2 + 16) / 4 = 17.06667 opt/byte
    
    public static float Im2colWinograd_64x64_intensity(int n, int r) { //cache block size = 64 * 64
	if (n == 3 && r == 2) return S4_intensity_f3x2;
	if (n == 2 && r == 3) return S4_intensity_f2x3;
        return -1;
    }
   
    public static float Im2colWinograd_64x32_intensity(int n, int r) { //cache block size = 64 * 32
	if (n == 7 && r == 2) return S8_intensity_f7x2;
	if (n == 6 && r == 3) return S8_intensity_f6x3;
	if (n == 5 && r == 4) return S8_intensity_f5x4;
	if (n == 4 && r == 5) return S8_intensity_f4x5;
	if (n == 3 && r == 6) return S8_intensity_f3x6;
	if (n == 2 && r == 7) return S8_intensity_f2x7;

	if (n == 8 && r == 5) return S8_intensity_f8x5;
	if (n == 6 && r == 6) return S8_intensity_f6x6;
	if (n == 4 && r == 7) return S8_intensity_f4x7;

        if (n ==  5 && r == 12) return S16_intensity_f5xC_c64;
	if (n ==  6 && r == 11) return S16_intensity_f6xB_c64;
	if (n ==  7 && r == 10) return S16_intensity_f7xA_c64;
	if (n ==  8 && r ==  9) return S16_intensity_f8x9_c64;
	if (n ==  9 && r ==  8) return S16_intensity_f9x8_c64;
        if (n == 10 && r ==  7) return S16_intensity_fAx7_c64;
        
        if (n == 6 && r == 2) return S4_intensity_f6x2;
        if (n == 4 && r == 3) return S4_intensity_f4x3;
	return -1;
    }
    
    public static float Im2colWinograd_32x32_intensity(int n, int r) {//cache block size = 32 * 32
        if (n ==  5 && r == 12) return S16_intensity_f5xC;
        if (n ==  6 && r == 11) return S16_intensity_f6xB;
        if (n ==  7 && r == 10) return S16_intensity_f7xA;
        if (n ==  9 && r ==  8) return S16_intensity_f8x9;
        if (n ==  8 && r ==  9) return S16_intensity_f9x8;
	if (n == 10 && r ==  7) return S16_intensity_fAx7;
	return -1;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="blockNum">
    //cache block size = 128 * 128
    public static final int GEMM_nblock(int OH, int OW, int N, int OC) {
        int GN = OC, GM = N * OH * OW;//get img2col size
        int bn = 0; for(;;) {
            if(GN > 127) bn += GN >> 7; GN &= 127; if(GN == 0) break;//2^7
            if(GN >  63) bn += 1;       GN &=  63; if(GN == 0) break;//2^6
            if(GN >  31) bn += 1;       GN &=  31; if(GN == 0) break;//2^5
            if(GN >  15) bn += 1;       GN &=  15; if(GN == 0) break;//2^4
            if(GN >   7) bn += 1;       GN &=   7; if(GN == 0) break;//2^3
            if(GN >   3) bn += 1; break;
        }
        int bm = 0; for(;;) {
            if(GM > 127) bm += GM >> 7; GM &= 127; if(GM == 0) break;//2^7
            if(GM >  63) bm += 1;       GM &=  63; if(GM == 0) break;//2^6
            if(GM >  31) bm += 1;       GM &=  31; if(GM == 0) break;//2^5
            if(GM >  15) bm += 1;       GM &=  15; if(GM == 0) break;//2^4
            if(GM >   7) bm += 1;       GM &=   7; if(GM == 0) break;//2^3
            if(GM >   3) bm += 1; break;
        }
        return bn * bm;
    }
    
    //cache block size = 128 * 128
    public static final int GEMMV2_nblock(int OH, int OW, int N, int OC)  {
        int GN = OC, GM = N;//get img2col size 
        int bn = 0; for(;;) {
            if(GN > 127) bn += GN >> 7; GN &= 127; if(GN == 0) break;//2^7
            if(GN >  63) bn += 1;       GN &=  63; if(GN == 0) break;//2^6
            if(GN >  31) bn += 1;       GN &=  31; if(GN == 0) break;//2^5
            if(GN >  15) bn += 1;       GN &=  15; if(GN == 0) break;//2^4
            if(GN >   7) bn += 1;       GN &=   7; if(GN == 0) break;//2^3
            if(GN >   3) bn += 1; break;
        }
        int bm = 0; for(;;) {
            if(GM > 127) bm += GM >> 7; GM &= 127; if(GM == 0) break;//2^7
            if(GM >  63) bm += 1;       GM &=  63; if(GM == 0) break;//2^6
            if(GM >  31) bm += 1;       GM &=  31; if(GM == 0) break;//2^5
            if(GM >  15) bm += 1;       GM &=  15; if(GM == 0) break;//2^4
            if(GM >   7) bm += 1;       GM &=   7; if(GM == 0) break;//2^3
            if(GM >   3) bm += 1; break;
        }
        return bn * bm * OH * OW;//the max stream size is 10
    }
    
    //cache block size = 64 * 64
    public static int Im2colWinograd_64x64_nblock(int OH, int OW, int N, int OC, int n) {//F(n, r)
	int GN = OC, GM = N; 
	int bn = 0; for (;;) {//OC % 64
		if (GN > 63) bn += GN >> 6; GN &= 63; if (GN == 0) break;//2^6
		if (GN > 31) bn += 1;       GN &= 31; if (GN == 0) break;//2^5
		if (GN > 15) bn += 1;       GN &= 15; if (GN == 0) break;//2^4
		if (GN >  7) bn += 1;       GN &=  7; if (GN == 0) break;//2^3
		if (GN >  3) bn += 1; break;
	}
	int bm = 0; for (;;) {//N % 64
		if (GM > 63) bm += GM >> 6; GM &= 63; if (GM == 0) break;//2^5
		if (GM > 31) bm += 1;       GM &= 31; if (GM == 0) break;//2^5
		if (GM > 15) bm += 1;       GM &= 15; if (GM == 0) break;//2^4
		if (GM >  7) bm += 1;       GM &=  7; if (GM == 0) break;//2^3
		if (GM >  3) bm += 1; break;
	}
	return bn * bm * OH * (OW / n);//the max stream size is 10
    }

    //cache block size = 64 * 32
    public static int Im2colWinograd_64x32_nblock(int OH, int OW, int N, int OC, int n) {//F(n, r)
	int GN = OC, GM = N;
	int bn = 0; for (;;) {//OC % 64
		if (GN > 63) bn += GN >> 6; GN &= 63; if (GN == 0) break;//2^6
		if (GN > 31) bn += 1;       GN &= 31; if (GN == 0) break;//2^5
		if (GN > 15) bn += 1;       GN &= 15; if (GN == 0) break;//2^4
		if (GN >  7) bn += 1;       GN &=  7; if (GN == 0) break;//2^3
		if (GN >  3) bn += 1; break;
	}
	int bm = 0; for (;;) {//N % 32
		if (GM > 31) bm += GM >> 5; GM &= 31; if (GM == 0) break;//2^5
		if (GM > 15) bm += 1;       GM &= 15; if (GM == 0) break;//2^4
		if (GM >  7) bm += 1;       GM &=  7; if (GM == 0) break;//2^3
		if (GM >  3) bm += 1; break;
	}
	return bn * bm * OH * (OW / n);//the max stream size is 10
    }

    //cache block size = 32 * 32
    public static int Im2colWinograd_32x32_nblock(int OH, int OW, int N, int OC, int n) {//F(n, r)
	int GN = OC, GM = N;
	int bn = 0; for (;;) {//OC % 32
		if (GN > 31) bn += GN >> 5; GN &= 31; if (GN == 0) break;//2^5
		if (GN > 15) bn += 1;       GN &= 15; if (GN == 0) break;//2^4
		if (GN >  7) bn += 1;       GN &=  7; if (GN == 0) break;//2^3
		if (GN >  3) bn += 1; break;
	}
	int bm = 0; for (;;) {//N % 32
		if (GM > 31) bm += GM >> 5; GM &= 31; if (GM == 0) break;//2^5
		if (GM > 15) bm += 1;       GM &= 15; if (GM == 0) break;//2^4
		if (GM >  7) bm += 1;       GM &=  7; if (GM == 0) break;//2^3
		if (GM >  3) bm += 1; break;
	}
	return bn * bm * OH * (OW / n);//the max stream size is 10
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="streamSize">
    public static final int max_streamsize = 10;
    
    public static final int nstream_SGM(int OH, int OW, int N, int OC, int GM_part) {
        int size = Cuda_conv3D.GEMM_nstream(OH, OW, N, OC) * GM_part;
        return (size < max_streamsize ? size : max_streamsize);//the max stream size is 10
    }
    
    //<editor-fold defaultstate="collapsed" desc="GEMM_nstream">
    public static final int GEMM_nstream(int OH, int OW, int N, int OC) {
        int GN = OC, GM = N * OH * OW;//get img2col size
        int sn = 0; if((GN & (GN - 1)) == 0) sn = 1; else for(;;) {
            if(GN > 127) sn++; GN &= 127; if(GN == 0) break;
            if(GN >  63) sn++; GN &=  63; if(GN == 0) break;
            if(GN >  31) sn++; GN &=  31; if(GN == 0) break;
            if(GN >  15) sn++; GN &=  15; if(GN == 0) break;
            if(GN >   7) sn++; GN &=   7; if(GN == 0) break;
            if(GN >   3) sn++; break;
        }
        int sm = 0; if((GM & (GM - 1)) == 0) sm = 1;  else for(;;) {
            if(GM > 127) sm++; GM &= 127; if(GM == 0) break;
            if(GM >  63) sm++; GM &=  63; if(GM == 0) break;
            if(GM >  31) sm++; GM &=  31; if(GM == 0) break;
            if(GM >  15) sm++; GM &=  15; if(GM == 0) break;
            if(GM >   7) sm++; GM &=   7; if(GM == 0) break;
            if(GM >   3) sm++; break;
        }
        int size = sn * sm;
        return (size < max_streamsize ? size : max_streamsize);//the max stream size is 10
    }
    
    public static final int GEMM_nstream(int GM, int OC) {
        int GN = OC;//get img2col size
        int sn = 0; if((GN & (GN - 1)) == 0) sn = 1; else for(;;) {
            if(GN > 127) sn++; GN &= 127; if(GN == 0) break;
            if(GN >  63) sn++; GN &=  63; if(GN == 0) break;
            if(GN >  31) sn++; GN &=  31; if(GN == 0) break;
            if(GN >  15) sn++; GN &=  15; if(GN == 0) break;
            if(GN >   7) sn++; GN &=   7; if(GN == 0) break;
            if(GN >   3) sn++; break;
        }
        int sm = 0; if((GM & (GM - 1)) == 0) sm = 1; else for(;;) {
            if(GM > 127) sm++; GM &= 127; if(GM == 0) break;
            if(GM >  63) sm++; GM &=  63; if(GM == 0) break;
            if(GM >  31) sm++; GM &=  31; if(GM == 0) break;
            if(GM >  15) sm++; GM &=  15; if(GM == 0) break;
            if(GM >   7) sm++; GM &=   7; if(GM == 0) break;
            if(GM >   3) sm++; break;
        }
        int size = sn * sm;
        return (size < max_streamsize ? size : max_streamsize);//the max stream size is 10
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Im2colWinograd_nstream">
    public static final int Im2colWinograd_s8_nstream(int FW, int OH, int OW, int N, int OC) {
        int OC64 = (OC & 63);
        int GM32 = ((N * OH) & 31) * OW;
        int size =  ((OC64 == 0 && GM32 == 0)? 1 : GEMM_nstream(GM32, OC64));
        if((size > 1) || (FW > 3)) return size;
        
        //size <= 1 && FW <= 3
        if((FW == 3) && (FW % 6 != 0)) return 2;
        if((FW == 2) && (FW % 7 != 0)) return 2;
        return size;
    }
    
    public static final int Im2colWinograd_s16_nstream(int FW, int OH, int OW, int N, int OC) {
        int OC31 = (OC & 31);
        int GM32 = ((N * OH) & 31) * OW;
        return ((OC31 == 0 && GM32 == 0)? 1 : GEMM_nstream(GM32, OC31));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="GEMMV2_nstream">
    public static final int GEMMV2_nstream(int N, int OC) {
        int GN = OC, GM = N;//get img2col size
        int sn = 0; if((GN & (GN - 1)) ==0 ) sn = 1; else for(;;) {
            if(GN > 127) sn++; GN &= 127; if(GN == 0) break;
            if(GN >  63) sn++; GN &=  63; if(GN == 0) break;
            if(GN >  31) sn++; GN &=  31; if(GN == 0) break;
            if(GN >  15) sn++; GN &=  15; if(GN == 0) break;
            if(GN >   7) sn++; GN &=   7; if(GN == 0) break;
            if(GN >   3) sn++; break;
        }
        int sm = 0; if((GM & (GM - 1)) == 0) sm = 1; else for(;;) {
            if(GM > 127) sm++; GM &= 127; if(GM == 0) break;
            if(GM >  63) sm++; GM &=  63; if(GM == 0) break;
            if(GM >  31) sm++; GM &=  31; if(GM == 0) break;
            if(GM >  15) sm++; GM &=  15; if(GM == 0) break;
            if(GM >   7) sm++; GM &=   7; if(GM == 0) break;
            if(GM >   3) sm++; break;
        }
        int size = sn * sm;
        return (size < max_streamsize ? size : max_streamsize);//the max stream size is 10
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="GM_part_slice">
    public static final int max_grid   = 65535;//65536 - 1, 2^16
    public static final int grid_slice = 32768;//2^15
    
    public static final int[] GEMM_GM_part_slice(int GN, int GM, int GK) {
        int max_GM = (max_grid + 1) << 6;
        if(GM < max_GM) return null;// no need to split GM, 419434 = 64 * 65536
        
        boolean flag = (GN > 127) && ((GK & 7) == 0);//GM >= 128
        int bm = (flag ? GM >> 7 : GM >> 6);//block_num along GM axis
        if(bm <= max_grid) return null;// no need to split GM
        
        int GM_part = bm >> 15;//bm / 32768
        int GM_slice = grid_slice << (flag ? 7 : 6);//128, 64
        return new int[] { GM_part, GM_slice };
    }
    
    public static final int GEMM_GM_slice(int GN, int GM, int GK) {
        int max_GM = (max_grid + 1) << 6;
        if(GM < max_GM) return -1;//no need to split GM, 419434 = 64 * 65536
        
        boolean flag = (GN > 127) && ((GK & 7) == 0);//GM >= 128
        int bm = (flag ? GM >> 7 : GM >> 6);//block_num along GM axis
        if(bm <= max_grid) return -1;//no need to split GM
        
        int GM_slice = grid_slice << (flag ? 7 : 6);//128, 64
        return GM_slice;
    }
    
    public static final int Im2col_Winograd_s8_GM_slice(int FW, int GM) {
        int n8 = 9 - FW, max_GM = ((max_grid + 1) << 5) * n8;
        if(GM < max_GM) return -1;// no need to split GM, 419434 = 64 * 65536
        return (grid_slice << 5) * n8;//GM_slice
    }
     
    public static final int Im2col_Winograd_s16_GM_slice(int FW, int GM) {
        int n16 = 17 - FW, max_GM = ((max_grid + 1) << 5) * n16;
        if(GM < max_GM) return -1;// no need to split GM, 419434 = 64 * 65536
        return (grid_slice << 5) * n16;//GM_slice
    }
    //</editor-fold>
 
    //======[Conv3D GEMM]=======================================================
    //<editor-fold defaultstate="collapsed" desc="conv3D">
    /**
     * <pre>
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) GM % 4 ==0, GM >= 4
     * (3) GN % 4 ==0, GN >= 4
     * (4) GK % 4 ==0, GK >= 8
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [4, 16]
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [4, 16]
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [8, 64]
     * (1) OC = 128: Size = 1.000000, Time = 1.478000 msec, Performance = 1452.965942 GFlop/s
     * (2) OC = 192: Size = 1.500000, Time = 2.297000 msec, Performance = 1402.362061 GFlop/s
     * (3) OC = 224: Size = 1.750000, Time = 3.024000 msec, Performance = 1242.756714 GFlop/s
     * (4) OC = 240: Size = 1.875000, Time = 3.783000 msec, Performance = 1064.375366 GFlop/s
     * (5) OC = 248: Size = 1.937500, Time = 4.354000 msec, Performance =  955.615417 GFlop/s
     * (6) OC = 252: Size = 1.968750, Time = 5.265000 msec, Performance =  803.012085 GFlop/s
     * 
     *  [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [32, 64]
     * (1) OC = 128: Size = 0.562500, Time = 0.912000 msec, Performance = 1324.517090 GFlop/s
     * (2) OC = 192: Size = 0.843750, Time = 1.389000 msec, Performance = 1304.491943 GFlop/s
     * (3) OC = 224: Size = 0.984375, Time = 1.812000 msec, Performance = 1166.627563 GFlop/s
     * (4) OC = 240: Size = 1.054688, Time = 2.267000 msec, Performance =  999.084351 GFlop/s
     * (5) OC = 248: Size = 1.089844, Time = 2.602000 msec, Performance =  899.470276 GFlop/s
     * (6) OC = 256: Size = 1.125000, Time = 1.691000 msec, Performance = 1428.692505 GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [4, 16]
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [8, 64]
     * (1) OC = 128: Size = 1.000000, Time = 1.925000 msec, Performance = 1115.575928 GFlop/s
     * (2) OC = 192: Size = 1.500000, Time = 3.140000 msec, Performance = 1025.867920 GFlop/s
     * (3) OC = 224: Size = 1.750000, Time = 4.220000 msec, Performance =  890.544189 GFlop/s
     * (4) OC = 240: Size = 1.875000, Time = 4.930000 msec, Performance =  816.740784 GFlop/s
     * (5) OC = 244: Size = 1.906250, Time = 5.820000 msec, Performance =  703.374695 GFlop/s
     * 
     * [IH, IW] = 32, [FH, FW] = 5, [sh, sw] = 2, [ph, pw] = 2, [N, IC] = [32, 64]
     * (1) OC = 128: Size = 1.562500, Time = 2.301000 msec, Performance = 1458.254272 GFlop/s
     * (2) OC = 192: Size = 2.343750, Time = 3.668000 msec, Performance = 1372.182373 GFlop/s
     * (3) OC = 224: Size = 2.734375, Time = 4.896000 msec, Performance = 1199.351685 GFlop/s
     * (4) OC = 240: Size = 2.929688, Time = 6.301000 msec, Performance =  998.485291 GFlop/s
     * (5) OC = 248: Size = 3.027344, Time = 7.447000 msec, Performance =  872.991943 GFlop/s
     * (6) OC = 256: Size = 3.125000, Time = 4.456000 msec, Performance = 1506.033813 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void conv3D(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="conv3D_texture">
    /**
     * <pre>
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) GM % 4 ==0, GM >= 4
     * (3) GN % 4 ==0, GN >= 4
     * (4) GK % 4 ==0, GK >= 8
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [4, 16]
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [4, 16]
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [8, 64]
     * (1) OC = 128: Size = 1.000000, Time = 1.476000 msec, Performance = 1454.934814 GFlop/s
     * (2) OC = 192: Size = 1.500000, Time = 2.307000 msec, Performance = 1396.283325 GFlop/s
     * (3) OC = 224: Size = 1.750000, Time = 2.999000 msec, Performance = 1253.116455 GFlop/s
     * (4) OC = 240: Size = 1.875000, Time = 3.741000 msec, Performance = 1076.325073 GFlop/s
     * (5) OC = 248: Size = 1.937500, Time = 4.314000 msec, Performance =  964.476013 GFlop/s
     * (6) OC = 252: Size = 1.968750, Time = 5.257000 msec, Performance =  804.234070 GFlop/s
     * 
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [32, 64]
     * (1) OC = 128: Size = 0.562500, Time = 0.903000 msec, Performance = 1337.718262 GFlop/s
     * (2) OC = 192: Size = 0.843750, Time = 1.375000 msec, Performance = 1317.774048 GFlop/s
     * (3) OC = 224: Size = 0.984375, Time = 1.791000 msec, Performance = 1180.306641 GFlop/s
     * (4) OC = 240: Size = 1.054688, Time = 2.239000 msec, Performance = 1011.578430 GFlop/s
     * (5) OC = 248: Size = 1.089844, Time = 2.598000 msec, Performance =  900.855103 GFlop/s
     * (6) OC = 256: Size = 1.125000, Time = 1.678000 msec, Performance = 1439.761108 GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [4, 16]
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [32, 64]
     * (1) OC = 128: Size = 0.562500, Time = 0.905000 msec, Performance = 1334.761963 GFlop/s
     * (2) OC = 192: Size = 0.843750, Time = 1.388000 msec, Performance = 1305.431763 GFlop/s
     * (3) OC = 224: Size = 0.984375, Time = 1.808000 msec, Performance = 1169.208618 GFlop/s
     * (4) OC = 240: Size = 1.054688, Time = 2.239000 msec, Performance = 1011.578430 GFlop/s
     * (5) OC = 248: Size = 1.089844, Time = 2.598000 msec, Performance =  900.855103 GFlop/s
     * (6) OC = 256: Size = 1.125000, Time = 1.681000 msec, Performance = 1437.191650 GFlop/s
     * 
     * [IH, IW] = 32, [FH, FW] = 5, [sh, sw] = 2, [ph, pw] = 2, [N, IC] = [32, 64]
     * (1) OC = 128: Size = 1.562500, Time = 2.270000 msec, Performance = 1478.168823 GFlop/s
     * (2) OC = 192: Size = 2.343750, Time = 3.625000 msec, Performance = 1388.459229 GFlop/s
     * (3) OC = 224: Size = 2.734375, Time = 4.766000 msec, Performance = 1232.065796 GFlop/s
     * (4) OC = 240: Size = 2.929688, Time = 6.032000 msec, Performance = 1043.013306 GFlop/s
     * (5) OC = 248: Size = 3.027344, Time = 7.162000 msec, Performance =  907.731201 GFlop/s
     * (6) OC = 256: Size = 3.125000, Time = 4.477000 msec, Performance = 1498.969360 GFlop/s
     * 
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void conv3D_texture(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="conv3D_np">
    /**
     * <pre>
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) GM % 4 ==0, GM >= 4
     * (3) GN % 4 ==0, GN >= 4
     * (4) GK % 4 ==0, GK >= 8
     * (5) ph = pw = 0
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 0, [N, IC] = [4, 16]
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 35, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 0, [N, IC] = [8, 64]
     * (1) OC = 128: Size = 1.000000, Time = 1.468000 msec, Performance = 1462.863525 GFlop/s
     * (2) OC = 192: Size = 1.500000, Time = 2.263000 msec, Performance = 1423.431519 GFlop/s
     * (3) OC = 224: Size = 1.750000, Time = 3.039000 msec, Performance = 1236.622681 GFlop/s
     * (4) OC = 240: Size = 1.875000, Time = 3.877000 msec, Performance = 1038.568970 GFlop/s
     * (5) OC = 248: Size = 1.937500, Time = 4.329000 msec, Performance =  961.134094 GFlop/s
     * (6) OC = 252: Size = 1.968750, Time = 4.963000 msec, Performance =  851.875610 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     */
    @Passed
    public static native void conv3D_np(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="conv3D_W1">
    /**
     *  <pre>
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH = FW = 1
     * (2) ph = pw = 0, sh = sw = 1
     * (2) GM % 4 ==0, GM >= 4
     * (2) GN % 4 ==0, GN >= 4
     * (3) GK % 4 ==0, GK >= 4
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [IH, IW] = 64, [FH, FW] = 1, [sh, sw] = 1, [ph, pw] = 0, [N, IC] = 4, 16
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 64, [FH, FW] = 1, [sh, sw] = 1, [ph, pw] = 0, [N, IC] = 8, 128
     * (1) OC = 128: Size = 0.500000, Time = 0.902000 msec, Performance = 1190.401123 GFlop/s
     * (2) OC = 192: Size = 0.750000, Time = 1.448000 msec, Performance = 1112.301636 GFlop/s
     * (3) OC = 224: Size = 0.875000, Time = 1.816000 msec, Performance = 1034.718140 GFlop/s
     * (4) OC = 240: Size = 0.937500, Time = 2.301000 msec, Performance =  874.952576 GFlop/s
     * (5) OC = 248: Size = 0.968750, Time = 2.602000 msec, Performance =  799.529114 GFlop/s
     * (6) OC = 252: Size = 0.984375, Time = 2.946000 msec, Performance =  717.559082 GFlop/s
     * (7) OC = 255: Size = 0.996094, Time = 1.634000 msec, Performance = 1309.115723 GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     *  [IH, IW] = 64, [FH, FW] = 1, [sh, sw] = 1, [ph, pw] = 0, [N, IC] = 4, 16
     * for OC from 1 to 128: correct
     * 
     * (1) OC = 128: Size = 0.250000, Time = 0.970000 msec, Performance = 553.475159 GFlop/s
     * (2) OC = 192: Size = 0.375000, Time = 1.430000 msec, Performance = 563.151306 GFlop/s
     * (3) OC = 252: Size = 0.492188, Time = 2.800000 msec, Performance = 377.487366 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param dY_address
     * @param N
     * @param IC
     * @param OC 
     */
    @Passed 
    public static native void conv3D_W1(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address,
            long dY_address,
            int N, int IC, int OC);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="conv3DV2">
    /**
     * <pre>
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) GM % 4 ==0, GM >= 4
     * (3) GN % 4 ==0, GN >= 4
     * (4) GK % 4 ==0, GK >= 8
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [4, 16]
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [4, 16]
     * for OC from 1 to 128: correct
     * 
     *  [IH, IW] = 8, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [512, 128]
     * (1) OC = 128: Size = 1.125000, Time = 1.655000 msec, Performance = 1459.769897 GFlop/s
     * (2) OC = 192: Size = 1.687500, Time = 2.750000 msec, Performance = 1317.774048 GFlop/s
     * (3) OC = 224: Size = 1.968750, Time = 3.727000 msec, Performance = 1134.386475 GFlop/s
     * (5) OC = 240: Size = 2.109375, Time = 4.608000 msec, Performance =  983.040039 GFlop/s
     * (8) OC = 256: Size = 2.250000, Time = 2.960000 msec, Performance = 1632.377808 GFlop/s
     * 
     * [IH, IW] = 4, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [512, 128]
     * (1) OC = 512: Size = 1.125000, Time = 1.328000 msec, Performance = 1819.216309 GFlop/s
     * (2) OC = 576: Size = 1.265625, Time = 1.634000 msec, Performance = 1663.347046 GFlop/s
     * (3) OC = 608: Size = 1.335938, Time = 1.890000 msec, Performance = 1517.938599 GFlop/s
     * (4) OC = 624: Size = 1.371094, Time = 2.135000 msec, Performance = 1379.110718 GFlop/s
     * (7) OC = 512, IC = 256: Size = 2.250000, Time = 2.521000 msec, Performance = 1916.635620 GFlop/s
     * 
     * [IH, IW] = 4, [FH, FW] = 4, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [512, 128]
     * (1) OC = 512: Size = 2.000000, Time = 1.667000 msec, Performance = 2576.465088 GFlop/s
     * (2) OC = 576: Size = 2.250000, Time = 2.141000 msec, Performance = 2256.813721 GFlop/s
     * (3) OC = 608: Size = 2.375000, Time = 2.511000 msec, Performance = 2031.172363 GFlop/s
     * (4) OC = 624: Size = 2.437500, Time = 2.867000 msec, Performance = 1825.772949 GFlop/s
     * 
     * [IH, IW] = 4, [FH, FW] = 5, [sh, sw] = 2, [ph, pw] = 2, [N, IC] = [512, 128]
     * (1) OC = 512: Size = 3.125000, Time = 2.624000 msec, Performance = 2557.502441 GFlop/s
     * (2) OC = 576: Size = 3.515625, Time = 3.402000 msec, Performance = 2219.208496 GFlop/s
     * (3) OC = 608: Size = 3.710938, Time = 4.004000 msec, Performance = 1990.303955 GFlop/s
     * (4) OC = 624: Size = 3.808594, Time = 4.563000 msec, Performance = 1792.437500 GFlop/s
     * 
     * </pre>
     * @param useTexture
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void conv3DV2(boolean useTexture, long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    
    //======[Conv3D GEMMR]======================================================
    //<editor-fold defaultstate="collapsed" desc="kernel_remode">
    /**
     * <pre>
     * Remode: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC].
     * </pre>
     * @param stream_address
     * @param dW_address
     * @param dCW_addres
     * @param FH
     * @param FW
     * @param OC
     * @param IC 
     */
    public static native void kernel_remode(long stream_address,
            long dW_address, long dCW_addres,
            int FH, int FW, int OC, int IC);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="conv3D_GemmR">
    /**
     * <pre>
     * Remode: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], CW[FH, FW, IC, OC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) GM % 4 ==0, GM >= 4
     * (3) GN % 4 ==0, GN >= 4
     * (4) GK % 4 ==0, GK >= 8
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [4, 16]
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [4, 16]
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [8, 64]
     * (1) OC = 128: Size = 1.000000, Time = 1.462000 msec, Performance = 1468.867065 GFlop/s
     * (2) OC = 192: Size = 1.500000, Time = 2.249000 msec, Performance = 1432.292236 GFlop/s
     * (3) OC = 224: Size = 1.750000, Time = 2.834000 msec, Performance = 1326.074829 GFlop/s
     * (4) OC = 240: Size = 1.875000, Time = 3.580000 msec, Performance = 1124.729614 GFlop/s
     * (5) OC = 248: Size = 1.937500, Time = 4.170000 msec, Performance =  997.781677 GFlop/s
     * (6) OC = 252: Size = 1.968750, Time = 5.093000 msec, Performance = 830.131226 GFlop/s
     * 
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [32, 64]
     * (1) OC = 128: Size = 0.562500, Time = 0.914000 msec, Performance = 1321.618774 GFlop/s
     * (2) OC = 192: Size = 0.843750, Time = 1.379000 msec, Performance = 1313.951660 GFlop/s
     * (3) OC = 224: Size = 0.984375, Time = 1.751000 msec, Performance = 1207.269653 GFlop/s
     * (4) OC = 240: Size = 1.054688, Time = 2.212000 msec, Performance = 1023.925964 GFlop/s
     * (5) OC = 248: Size = 1.089844, Time = 2.601000 msec, Performance =  899.816040 GFlop/s
     * (6) OC = 256: Size = 1.125000, Time = 1.679000 msec, Performance = 1438.903564 GFlop/s
     * 
     * [IH, IW] = 32, [FH, FW] = 5, [sh, sw] = 2, [ph, pw] = 2, [N, IC] = [32, 64]
     * (1) OC = 128: Size = 1.562500, Time = 2.228000 msec, Performance = 1506.033813 GFlop/s
     * (2) OC = 192: Size = 2.343750, Time = 3.496000 msec, Performance = 1439.692383 GFlop/s
     * (3) OC = 224: Size = 2.734375, Time = 4.488000 msec, Performance = 1308.383667 GFlop/s
     * (4) OC = 240: Size = 2.929688, Time = 5.754000 msec, Performance = 1093.405640 GFlop/s
     * (5) OC = 248: Size = 3.027344, Time = 6.851000 msec, Performance =  948.937561 GFlop/s
     * (6) OC = 256: Size = 3.125000, Time = 4.385000 msec, Performance = 1530.418701 GFlop/s
     * 
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param dCW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    protected static native void conv3D_GemmR(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, long dCW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="conv3D_GemmR_SGM">
    /**
     * <pre>
     * Remode: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], CW[FH, FW, IC, OC], [sh, sw], [ph, pw]).
     * Split 1 grid to multi grads along GM axis.
     * (1) FH * FW >= 2
     * (2) GM % 4 ==0, GM >= 4
     * (3) GN % 4 ==0, GN >= 4
     * (4) GK % 4 ==0, GK >= 8
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param dCW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     * @param GM_slice 
     */
    @Passed
    protected static native void conv3D_GemmR_SGM(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, long dCW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw, 
            int GM_slice);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="integration">
    public static final void CONV3D_GemmR(long[] streamArray, int length,
            long X_address, int IH, int IW,
            long W_address, long CW_address, int FH, int FW,
            long Y_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        int GM = N * OH * OW;
        int GK = FH * FW * IC;
        int GM_slice = GEMM_GM_slice(OC, GM, GK); 
        if(GM_slice < 0) {
            conv3D_GemmR(streamArray, length,
                    X_address, IH, IW, 
                    W_address, CW_address, FH, FW,
                    Y_address, OH, OW, 
                    N, IC, OC, 
                    sh, sw, ph, pw);
        }
        else {
            conv3D_GemmR_SGM(streamArray, length,
                    X_address, IH, IW,
                    W_address, CW_address, FH, FW,
                    Y_address, OH, OW,
                    N, IC, OC,
                    sh, sw, ph, pw,
                    GM_slice);
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="conv3D_GemmR_texture">
    /**
     * <pre>
     * Remode: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], CW[FH, FW, IC, OC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) GM % 4 ==0, GM >= 4
     * (3) GN % 4 ==0, GN >= 4
     * (4) GK % 4 ==0, GK >= 8
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]------------
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [4, 16]
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [4, 16]
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [64, 64]
     * (1) OC = 128: Size = 1.125000, Time = 1.603000 msec, Performance = 1507.123535 GFlop/s
     * (2) OC = 192: Size = 1.687500, Time = 2.421000 msec, Performance = 1496.851929 GFlop/s
     * (3) OC = 224: Size = 1.968750, Time = 2.996000 msec, Performance = 1411.167725 GFlop/s
     * (4) OC = 240: Size = 2.109375, Time = 3.662000 msec, Performance = 1236.987549 GFlop/s
     * (5) OC = 248: Size = 2.179688, Time = 4.187000 msec, Performance = 1117.946899 GFlop/s
     * (6) OC = 252: Size = 2.214844, Time = 4.750000 msec, Performance = 1001.334900 GFlop/s
     * (7) OC = 256: Size = 2.250000, Time = 3.051000 msec, Performance = 1583.689941 GFlop/s
     * => OC =  64: Size = 0.562500, Time = 1.035000 msec, Performance = 1167.110718 GFlop/s
     * => OC = 136: Size = 1.195313, Time = 2.186000 msec, Performance = 1174.251587 GFlop/s
     * => OC = 144: Size = 1.265625, Time = 2.225000 msec, Performance = 1221.532227 GFlop/s
     * => OC = 160: Size = 1.406250, Time = 2.242000 msec, Performance = 1346.966431 GFlop/s
     * 
     * [IH, IW] = 32, [FH, FW] = 4, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [32, 64]
     * (1) OC = 128: Size = 1.000000, Time = 1.459000 msec, Performance = 1471.887329 GFlop/s
     * (2) OC = 192: Size = 1.500000, Time = 2.214000 msec, Performance = 1454.934692 GFlop/s
     * (3) OC = 224: Size = 1.750000, Time = 2.795000 msec, Performance = 1344.578247 GFlop/s
     * (4) OC = 240: Size = 1.875000, Time = 3.274000 msec, Performance = 1229.850952 GFlop/s
     * (5) OC = 248: Size = 1.937500, Time = 3.774000 msec, Performance = 1102.477417 GFlop/s
     * (6) OC = 252: Size = 1.968750, Time = 4.275000 msec, Performance =  988.972717 GFlop/s
     * 
     * [IH, IW] = 32, [FH, FW] = 5, [sh, sw] = 2, [ph, pw] = 2, [N, IC] = [32, 64]
     * (1) OC = 128: Size = 1.562500, Time = 2.226000 msec, Performance = 1507.386841 GFlop/s
     * (2) OC = 192: Size = 2.343750, Time = 3.303000 msec, Performance = 1523.816162 GFlop/s
     * (3) OC = 224: Size = 2.734375, Time = 4.160000 msec, Performance = 1411.544678 GFlop/s
     * (4) OC = 240: Size = 2.929688, Time = 5.049000 msec, Performance = 1246.079712 GFlop/s
     * (5) OC = 248: Size = 3.027344, Time = 5.759000 msec, Performance = 1128.871582 GFlop/s
     * (6) OC = 256: Size = 3.125000, Time = 4.230000 msec, Performance = 1586.497925 GFlop/s
     * 
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param dCW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    protected static native void conv3D_GemmR_texture(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, long dCW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="conv3D_GemmR_texture_SGM">
    /**
     * <pre>
     * Remode: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], CW[FH, FW, IC, OC], [sh, sw], [ph, pw]).
     * Split 1 grid to multi grads along GM axis.
     * (1) FH * FW >= 2
     * (2) GM % 4 ==0, GM >= 4
     * (3) GN % 4 ==0, GN >= 4
     * (4) GK % 4 ==0, GK >= 8
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param dCW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     * @param GM_slice 
     */
    @Passed
    protected static native void conv3D_GemmR_texture_SGM(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, long dCW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw,
            int GM_slice);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="integration">
    public static final void CONV3D_GemmR_texture(long[] streamArray, int length,
            long X_address, int IH, int IW,
            long W_address, long CW_address, int FH, int FW,
            long Y_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        int GM = N * OH * OW;
        int GK = FH * FW * IC;
        int GM_slice = GEMM_GM_slice(OC, GM, GK); 
        if (GM_slice < 0) {
            conv3D_GemmR_texture(streamArray, length,
                    X_address, IH, IW,
                    W_address, CW_address, FH, FW,
                    Y_address, OH, OW,
                    N, IC, OC,
                    sh, sw, ph, pw);
        } 
        else {
            conv3D_GemmR_texture_SGM(streamArray, length,
                    X_address, IH, IW,
                    W_address, CW_address, FH, FW,
                    Y_address, OH, OW,
                    N, IC, OC,
                    sh, sw, ph, pw,
                    GM_slice);
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="conv3D_W1">
    /**
     * <pre>
     * Remode: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH = FW = 1
     * (2) ph = pw = 0, sh = sw = 1
     * (2) GM % 4 ==0, GM >= 4
     * (2) GN % 4 ==0, GN >= 4
     * (3) GK % 4 ==0, GK >= 4
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]------------
     * [IH, IW] = 64, [FH, FW] = 1, [sh, sw] = 1, [ph, pw] = 0, [N, IC] = 4, 16
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 64, [FH, FW] = 1, [sh, sw] = 1, [ph, pw] = 0, [N, IC] = 8, 128
     * (1) OC = 128: Size = 0.500000, Time = 0.871000 msec, Performance = 1232.769043 GFlop/s
     * (2) OC = 192: Size = 0.750000, Time = 1.202000 msec, Performance = 1339.943970 GFlop/s
     * (3) OC = 224: Size = 0.875000, Time = 1.580000 msec, Performance = 1189.270996 GFlop/s
     * (4) OC = 240: Size = 0.937500, Time = 2.059000 msec, Performance =  977.788208 GFlop/s
     * (5) OC = 248: Size = 0.968750, Time = 2.412000 msec, Performance =  862.510315 GFlop/s
     * (6) OC = 252: Size = 0.984375, Time = 2.683000 msec, Performance =  787.897583 GFlop/s
     * (7) OC = 255: Size = 0.996094, Time = 1.892000 msec, Performance = 1130.599976 GFlop/s
     * (8) OC = 256: Size = 1.000000, Time = 1.560000 msec, Performance = 1376.592163 GFlop/s
     * 
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param dCW_address
     * @param dY_address
     * @param N
     * @param IC
     * @param OC 
     */
    @Passed 
    protected static native void conv3D_GemmR_W1(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, long dCW_address,
            long dY_address,
            int N, int IC, int OC);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="conv3D_W1_SGM">
    /**
     * <pre>
     * Remode: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * Split 1 grid to multi grads along GM axis.
     * (1) FH = FW = 1
     * (2) ph = pw = 0, sh = sw = 1
     * (2) GM % 4 ==0, GM >= 4
     * (2) GN % 4 ==0, GN >= 4
     * (3) GK % 4 ==0, GK >= 4
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param dCW_address
     * @param dY_address
     * @param N
     * @param IC
     * @param OC 
     * @param GM_slice 
     */
    @Passed 
    protected static native void conv3D_GemmR_W1_SGM(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, long dCW_address,
            long dY_address,
            int N, int IC, int OC,
            int GM_slice);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="integration">
    public static final void CONV3D_GemmR_W1(long[] streamArray, int length,
            long X_address, int IH, int IW,
            long W_address, long CW_address,//OH = IH, OW = IW
            long Y_address,
            int N, int IC, int OC)
    {
        int GM = N * IH * IW;
        int GK = IC;
        int GM_slice = GEMM_GM_slice(OC, GM, GK);
        if (GM_slice < 0) {
            conv3D_GemmR_W1(streamArray, length,
                    X_address, IH, IW,
                    W_address, CW_address,
                    Y_address,
                    N, IC, OC);
        } 
        else {
            conv3D_GemmR_W1_SGM(streamArray, length,
                    X_address, IH, IW,
                    W_address, CW_address,
                    Y_address,
                    N, IC, OC,
                    GM_slice);
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="conv3D_GemmV2R">
    /**
     * <pre>
     * Remode: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], CW[FH, FW, IC, OC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) GM % 4 ==0, GM >= 4
     * (3) GN % 4 ==0, GN >= 4
     * (4) GK % 4 ==0, GK >= 8
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [4, 16]
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [4, 16]
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 8, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [512, 128]
     * (1) OC = 128: Size = 1.125000, Time = 1.548000 msec, Performance = 1560.671265 GFlop/s
     * (2) OC = 192: Size = 1.687500, Time = 2.397000 msec, Performance = 1511.839233 GFlop/s
     * (3) OC = 224: Size = 1.968750, Time = 3.294000 msec, Performance = 1283.502930 GFlop/s
     * (5) OC = 240: Size = 2.109375, Time = 4.249000 msec, Performance = 1066.097534 GFlop/s
     * (6) OC = 248: Size = 2.179688, Time = 5.380000 msec, Performance =  870.045227 GFlop/s
     * (7) OC = 252: Size = 2.214844, Time = 6.715000 msec, Performance =  708.315796 GFlop/s
     * (8) OC = 256: Size = 2.250000, Time = 2.687000 msec, Performance = 1798.227783 GFlop/s
     * 
     * [IH, IW] = 4, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [512, 128]
     * (1) OC = 512: Size = 1.125000, Time = 1.334000 msec, Performance = 1811.033813 GFlop/s
     * (2) OC = 576: Size = 1.265625, Time = 1.551000 msec, Performance = 1752.359131 GFlop/s
     * (3) OC = 608: Size = 1.335938, Time = 1.826000 msec, Performance = 1571.141235 GFlop/s
     * (4) OC = 624: Size = 1.371094, Time = 2.066000 msec, Performance = 1425.170044 GFlop/s
     * (5) OC = 632: Size = 1.388672, Time = 2.292000 msec, Performance = 1301.112549 GFlop/s
     * (6) OC = 636: Size = 1.397461, Time = 2.612000 msec, Performance = 1148.937378 GFlop/s
     * (7) OC = 512, IC = 256: Size = 2.250000, Time = 2.521000 msec, Performance = 1916.635620 GFlop/s
     * 
     * [IH, IW] = 4, [FH, FW] = 4, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [512, 128]
     * (1) OC = 512: Size = 2.000000, Time = 1.878000 msec, Performance = 2286.989990 GFlop/s
     * (2) OC = 576: Size = 2.250000, Time = 2.277000 msec, Performance = 2122.019531 GFlop/s
     * (3) OC = 608: Size = 2.375000, Time = 2.576000 msec, Performance = 1979.919922 GFlop/s
     * (4) OC = 624: Size = 2.437500, Time = 2.886000 msec, Performance = 1813.753174 GFlop/s
     * (5) OC = 632: Size = 2.468750, Time = 3.270000 msec, Performance = 1621.284546 GFlop/s
     * (6) OC = 636: Size = 2.484375, Time = 3.800000 msec, Performance = 1403.988037 GFlop/s
     * 
     * [IH, IW] = 4, [FH, FW] = 5, [sh, sw] = 2, [ph, pw] = 2, [N, IC] = [512, 128]
     * (1) OC = 512: Size = 3.125000, Time = 2.724000 msec, Performance = 2463.614746 GFlop/s
     * (2) OC = 576: Size = 3.515625, Time = 3.384000 msec, Performance = 2231.012695 GFlop/s
     * (3) OC = 608: Size = 3.710938, Time = 3.867000 msec, Performance = 2060.816406 GFlop/s
     * (4) OC = 624: Size = 3.808594, Time = 4.397000 msec, Performance = 1860.107544 GFlop/s
     * (5) OC = 632: Size = 3.857422, Time = 4.953000 msec, Performance = 1672.471313 GFlop/s
     * (6) OC = 636: Size = 3.881836, Time = 5.837000 msec, Performance = 1428.161621 GFlop/s
     * 
     * </pre>
     * @param useTexture
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param dCW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void conv3D_GemmV2R(boolean useTexture, long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, long dCW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    
    //======[Conv3D Im2col-WinogradR]===========================================
    //<editor-fold defaultstate="collapsed" desc="conv3D_Im2col_Winograd_s8_R_texture">
    /**
     * <pre>
     * Remode: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
     * Im2col-Winograd(n, r): n + r = 8 + 1.
     * </pre>
     * @param useTexture
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param dCW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param ph
     * @param pw 
     * @return  
     */
    @Passed
    protected static native boolean conv3D_Im2col_Winograd_s8_R_texture(boolean useTexture, long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, long dCW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="conv3D_Im2col_Winograd_s8_R_texture_SGM">
    /**
     * <pre>
     * Remode: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
     * Im2col-Winograd(n, r): n + r = 8 + 1.
     * Split 1 grid to multi grads along GM axis.
     * (1) FH * FW >= 2
     * (2) GM % 4 ==0, GM >= 4
     * (3) GN % 4 ==0, GN >= 4
     * (4) GK % 4 ==0, GK >= 8
     * </pre>
     * @param useTexture
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param dCW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param ph
     * @param pw 
     * @param GM_slice 
     * @return  
     */
    @Passed
    protected static native boolean conv3D_Im2col_WinogradR_s8_texture_SGM(boolean useTexture, long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, long dCW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int ph, int pw, 
            int GM_slice);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="integration">
    public static final boolean CONV3D_Im2col_Winograd_s8_R_texture(boolean useTexture, long[] streamArray, int length,
            long X_address, int IH, int IW,
            long W_address, long CW_address, int FH, int FW,
            long Y_address, int OH, int OW,
            int N, int IC, int OC,
            int ph, int pw)
    {
        int GM = N * OH * OW;
        int GM_slice = Im2col_Winograd_s8_GM_slice(FW, GM);
        if(GM_slice < 0) {
            return conv3D_Im2col_Winograd_s8_R_texture(useTexture, streamArray, length, 
                    X_address, IH, IW, 
                    W_address, CW_address, FH, FW, 
                    Y_address, OH, OW,
                    N, IC, OC, 
                    ph, pw);
        }
        else {
            return conv3D_Im2col_WinogradR_s8_texture_SGM(useTexture, streamArray, length, 
                    X_address, IH, IW,
                    W_address, CW_address, FH, FW, 
                    Y_address, OH, OW,
                    N, IC, OC, 
                    ph, pw, 
                    GM_slice);
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="conv3D_Im2col_Winograd_s16_R_texture">
    /**
     * <pre>
     * Remode: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
     * Im2col-Winograd(n, r): n + r = 16 + 1.
     * </pre>
     * @param useTexture
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param dCW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param ph
     * @param pw 
     * @return  
     */
    @Passed
    protected static native boolean conv3D_Im2col_Winograd_s16_R_texture(boolean useTexture, long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, long dCW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="conv3D_Im2col_Winograd_s16_R_texture_SGM">
    /**
     * <pre>
     * Remode: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
     * Im2col-Winograd(n, r): n + r = 16 + 1.
     * </pre>
     * @param useTexture
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param dCW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param ph
     * @param pw 
     * @param GM_slice 
     * @return  
     */
    @Passed
    protected static native boolean conv3D_Im2col_WinogradR_s16_texture_SGM(boolean useTexture, long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, long dCW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int ph, int pw, 
            int GM_slice);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="integration">
    public static final boolean CONV3D_Im2col_Winograd_s16_R_texture(boolean useTexture, long[] streamArray, int length,
            long X_address, int IH, int IW,
            long W_address, long CW_address, int FH, int FW,
            long Y_address, int OH, int OW,
            int N, int IC, int OC,
            int ph, int pw)
    {
        int GM = N * OH * OW;
        int GM_slice = Im2col_Winograd_s16_GM_slice(FW, GM);
        if(GM_slice < 0) {
            return conv3D_Im2col_Winograd_s16_R_texture(useTexture, streamArray, length,
                    X_address, IH, IW, 
                    W_address, CW_address, FH, FW,
                    Y_address, OH, OW,
                    N, IC, OC,
                    ph, pw);
        }
        else {
            return conv3D_Im2col_WinogradR_s16_texture_SGM(useTexture, streamArray, length,
                    X_address, IH, IW, 
                    W_address, CW_address, FH, FW,
                    Y_address, OH, OW,
                    N, IC, OC,
                    ph, pw,
                    GM_slice);
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Document">
    /**
     * <pre>
     * 3D Convolution. 
     * X(N, IH, IW, IC): the input Tensor
     * Y(N, OH, OW, OC): the output tensor
     * W(OC, FH, FW, IC): a 4D covolution kernel
     * (1) N: the batch size, how many samples in this batch
     * (2) IC: input channel number of X.
     *  => each element of X is a 3D tensor(Z: IC, X: IH, Y: IW), like a colorful pitcure
     * (3) OC: output channel number of Y.
     *  => each element of Y is a 3D tensor(Z: OC, X: OH, Y: OW)
     * (4) each element of W is a 3D tensor(a 3D convlution kernel), with its dimension(Z: IC, Y: FH, X: FW)
     * (5) Padding:
     *  => ph: the padding on Y axis
     *  => pw: the padding on X axis
     * (5) Stride: to move the kernel on each element of X
     *  => sh: the step on Y axis
     *  => sw: the step on X axis
     * (6) OH = (IH + 2*ph - FH)/sh + 1
     *     OW = (IW + 2*pw - FW)/sw + 1
     *  kindly let: (IH + 2*ph - FH)%sh==0, (IW + 2*pw - FW)%sw == 0,
     *  if not the result is also correct, but may has some bad effect on backprop of CNN.
     *
     * Use "Img2col Algorithm" to implement the 3D convolution as an implicit Matrix Multiply:
     * (1) GN = OC;
     * (2) GM = N * OH * OW;
     * (3) GK = IC * FH * FW;
     * W -> Matrix A[GN, GK] = [OC, IC*FH*FW(the size of each patch on X)]
     * X -> Matrix B[GK, GM] = [IC*FH*FW, N*OH*OW(the times of patchs on X)]
     * Y -> Matrix C[GN, GM] = [OC, N*OH*OW]
     * Y = conv3D(X, W) <-> C = A*B
     *
     * The img2co algorithm js just implemented logically, by a specific way to fetch  memory, not physically.
     * Make Sure: <b>
     * (1) FH>=2, FW>=2
     * (2) GN%4==0, GN>4
     * (3) GM%4==0, GM>4
     * (4) GK>=4</b>
     * </pre>
     */
    public static final boolean IM2COL_GEMM = true;
    //</editor-fold>
}
