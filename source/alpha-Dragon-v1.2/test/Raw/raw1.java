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
public class raw1 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    public static byte[] process_bytes(byte[] X)
    {
        byte[] Y = new byte[X.length >> 1]; int index = 0;
        int max0 = 0;
        int max1 = 0;
        for(int i=0; i<X.length; i+=2)
        {
            int b0 = X[i    ] & 0xff;
            int b1 = X[i + 1] & 0xff;
            int y = ((b0 << 8) + b1) >> 4;
            Y[index++] = (byte) y; 
            
            if(max0 < b0) max0 = b0;
            if(max1 < b1) max1 = b1;
        }
        
        System.out.println(max0 + ", " + max1);
        
        return Y;
    }
    
    public static byte[] process_bytes2(byte[] X)
    {
        byte[] Y = new byte[X.length >> 1]; 
        for(int i=0; i<Y.length; i++)
        {
            int b0 = X[i] & 0xff;
            int b1 = X[i] & 0xff;
//            int y = b0 | b1;
            Y[i] = (byte) b0;
        }
        
        return Y;
    }
    
    public static void test1() throws IOException
    {
        File f = new File("C:\\Users\\Gilgamesh\\Desktop\\a.raw");
        System.out.println(f.length());
          
        byte[] bytes =  process_bytes(fl.to_bytes(f));
        
        int IH = 640, IW = 693, IC = 224; 
//        Tensor X = eg.tensor_int8(bytes, IC, IH, IW)
//                .img().pixel_to_dtype(true)
//                .transpose(true, 0, 2);
//
        Tensor X = eg.tensor_int8(bytes, IH, IW, IC)
                .img().pixel_to_dtype(true);

        System.out.println(X);
          
        Tensor gray = eg.row_mean(X, X.lastDim())
                .img().dtype_to_pixel(true);
        
        BufferedImage img = cv.gray(gray.value_int8(), IH, IW); cv.imshow(img);
        X.delete();
    }
    
    public static byte[] sub_channel(byte[] X, int IH, int IW, int IC, int idx)
    {
        byte[] Y = new byte[IH * IW];
        for(int ih=0; ih<IH; ih++)
        for(int iw=0; iw<IW; iw++)
        for(int ic=0; ic<IC; ic++)
        {
            int xindex = (ic*IH + ih)*IW + iw;//(ih, iw, ic)
//            int xindex = (ih*IW + iw)*IC + ic;//(ih, iw, ic)
            int yindex = ih*IW + iw;//(ih, iw)
            Y[yindex] = X[xindex];
        }
        return Y;
    }
    
    public static void test2() throws IOException
    {
        File f = new File("C:\\Users\\Gilgamesh\\Desktop\\a.raw");
        System.out.println(f.length());
          
        byte[] bytes =  process_bytes(fl.to_bytes(f));
        
        int IH = 640, IW = 693, IC = 224; 
        for(int i=100; i<105; i++)
        {
            byte[] sub = sub_channel(bytes, IH, IW, IC, i);
            BufferedImage img = cv.gray(sub, IH, IW); cv.imshow(img);
        }
    }
    
    public static void main(String[] args)
    {
        try
        {
             test1();
        }
        catch(IOException e) {
            e.printStackTrace();
        }
    }
}
