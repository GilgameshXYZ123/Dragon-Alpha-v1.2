/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Img;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import static z.dragon.alpha.Alpha.alpha;
import static z.dragon.common.DragonCV.cv;
import z.dragon.common.DragonFile;
import static z.dragon.common.DragonFile.fl;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;

/**
 *
 * @author Gilgamesh
 */
public class Raw1 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    public static byte[][][] to3D(byte[] X, int IH, int IW, int IC)
    {
        byte[][][] img = new byte[IH][IW][IC]; int index = 0;
        for(int ih=0; ih<IH; ih++)
        for(int iw=0; iw<IW; iw++)
        for(int ic=0; ic<IC; ic++) {
            img[ih][iw][ic] = X[index++];
        }
        return img;
    }
     
    public static byte[] to1D(byte[][][] X, int IH, int IW, int IC)
    {
        byte[] img = new byte[IH*IW*IC]; int index = 0;
        for(int ih=0; ih<IH; ih++)
        for(int iw=0; iw<IW; iw++)
        for(int ic=0; ic<IC; ic++) {
            img[index++] = X[ih][iw][ic];
        }
        return img;
    }
    
    public static byte[][][] get3channel(byte[][][] X, int IH, int IW, int c1, int c2, int c3) {
        byte[][][] y = new byte[IH][IW][3];
        for(int ih=0; ih<IH; ih++)
        for(int iw=0; iw<IW; iw++)
        {
            y[ih][iw][0] = X[ih][iw][c1];
            y[ih][iw][1] = X[ih][iw][c2];
            y[ih][iw][2] = X[ih][iw][c3];
        }
        return y;
    }
    
    public static void test1() throws IOException
    {
        File f = new File("C:\\Users\\Gilgamesh\\Desktop\\a.raw");
        System.out.println(f.length());
       
        byte[] pixels = fl.to_bytes(f);
        Tensor X = eg.tensor_int8(pixels, 224, 880, 880);
        System.out.println(pixels.length);
        
        Tensor fX = X.img().pixel_to_dtype(true).transpose(true, 0, 2);
        Tensor gray = eg.row_mean(fX, fX.lastDim()).img().dtype_to_pixel(true);
        
        BufferedImage img = cv.gray(gray.value_int8(), 880, 880); cv.imshow(img);
    }
    
    public static void test2() throws IOException
    {
        File f = new File("C:\\Users\\Gilgamesh\\Desktop\\a.raw");
        System.out.println(f.length());
       
        byte[] bytes = fl.to_bytes(f);
        byte[][][] pixels = to3D(bytes, 880, 880, 224);
        byte[][][] pixels2 = get3channel(pixels, 880, 880, 13, 76, 132);
        byte[] bytes2 = to1D(pixels2, 880, 880, 3);
        
        System.out.println(bytes2.length);
        
        BufferedImage img = cv.BGR(bytes2, 880, 880); cv.imshow(img);
    }

    public static void test3() throws IOException
    {
        File f = new File("C:\\Users\\Gilgamesh\\Desktop\\a.raw");
        System.out.println(f.length());
        
        char[] chars = fl.to_chars(f);
        System.out.println("chars = " + chars.length);
    }
    
    public static void test4() throws IOException
    {
        File f = new File("C:\\Users\\Gilgamesh\\Desktop\\a.raw");
        System.out.println(f.length());
        
        byte[] bytes = fl.to_bytes(f);
        byte[] pixel = new byte[bytes.length >> 1]; int index = 0;
        int max = 0;//2539
        for(int i=0; i<bytes.length; i += 2) {
            int b0 = bytes[i    ] & 0xff;
            int b1 = bytes[i + 1] & 0xff;
            int b = ((b1 << 8) + b0) >> 4;
            if(max < b1) max = b1;
            pixel[index++] = (byte) b0;
        }
        
        System.out.println("max = " + max);
        
        byte[][][] raw = to3D(bytes, 640, 605, 224);
        byte[][][] rgb = get3channel(raw, 640, 605, 13, 76, 132);
        byte[] pixels = to1D(rgb, 640, 605, 3);
        BufferedImage img = cv.BGR(pixels, 640, 605); cv.imshow(img);
    }
    
    
    public static void main(String[] args)
    {
        try
        {
            test4();
        }
        catch(IOException e) {
            e.printStackTrace();
        }
    }
}
