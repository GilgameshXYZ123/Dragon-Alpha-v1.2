/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Raw;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import static z.dragon.alpha.Alpha.alpha;
import static z.dragon.common.DragonCV.cv;
import static z.dragon.common.DragonFile.fl;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;

/**
 *
 * @author Gilgamesh
 */
public class raw2 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static String path = "C:\\Users\\Gilgamesh\\Desktop\\a.raw";
    
    public static byte[] process_bytes(byte[] X)
    {
        byte[] Y = new byte[X.length >> 1]; int index = 0;//16 bit -> 8bit
        int max = 0;
        int maxy = 0;
        for(int i=0; i<X.length; i+=2)
        {
            int b0 = X[i    ] & 0xff;
            int b1 = X[i + 1] & 0xff;
            int b = (b1 << 8) + b0;
            int y = b >> 4;
            Y[index++] = (byte) y;
        }
        
        return Y;
    }
    
    public static void test1() throws IOException
    {
        File f = new File(path);
        System.out.println(f.length());
          
        byte[] bytes = process_bytes(fl.to_bytes(f));
        int IH = 640, IW = 693, IC = 224;
        
//        Tensor X = eg.tensor_int8(bytes, IH, IW, IC);//[IH, IW, IC]
//        Tensor X = eg.tensor_int8(bytes, IC, IH, IW) //[IC, IH, IW]
//                .img().pixel_to_dtype(true)
//                .transpose(true, 0, 2)
//                .img().dtype_to_pixel(true);

//        Tensor X = eg.tensor_int8(bytes, IH, IC, IW) //[IH, IC, IW]
//                .img().pixel_to_dtype(true)
//                .transpose(true, 1, 2)
//                .img().dtype_to_pixel(true);

//        Tensor X = eg.tensor_int8(bytes, IC, IW, IH) //[IC, IW, IH]
//                .img().pixel_to_dtype(true)
//                .transpose(true, 1, 2)
//                .transpose(true, 0, 2)
//                .img().dtype_to_pixel(true);

        Tensor X = eg.tensor_int8(bytes, IW, IC, IH) //[IW, IC, IH]
                .img().pixel_to_dtype(true)
                .transpose(true, 0, 2)
                .transpose(true, 1, 2)
                .img().dtype_to_pixel(true);
        
        System.out.println(X);

        
        
        Tensor rgb = X.img().extract_3channels(false, 13, 80, 132);
        
        Tensor fX = X.img().pixel_to_dtype(true);
        Tensor gray = eg.row_mean(fX, fX.lastDim())
                .img().dtype_to_pixel(true);
        
        BufferedImage img1 = cv.gray(gray.pixel(), IH, IW); cv.imshow(img1, "img1");
        BufferedImage img2 = cv.RGB(rgb.pixel(), IH, IW); cv.imshow(img2, "img2");
       
        System.gc();
    }
    
    public static void main(String[] args)
    {
        try
        {
            test1();
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }
}
