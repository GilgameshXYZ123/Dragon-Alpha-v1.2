/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Img;

import java.awt.image.BufferedImage;
import java.io.IOException;
import static z.dragon.alpha.Alpha.alpha;
import static z.dragon.common.DragonCV.cv;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;
import z.util.lang.SimpleTimer;

/**
 *
 * @author Gilgamesh
 */
public class AT4 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static String path = "C:\\Users\\Gilgamesh\\Desktop\\sj22.jpg";
    
    public static byte pix_restrict(float v) {
        if(v < 0) return 0;
        if(v > 255) return (byte) 255;
        return (byte) v;
    }
    
    public static byte[] resize(byte[] pixel, 
            int IH, int IW, int C, 
            int OH, int OW) 
    {
        byte[] pixel2 = new byte[OH * OW * 3];
        for(int oh=0; oh<OH; oh++)
        for(int ow=0; ow<OW; ow++)
        {
            int ih = (int) (Math.round(oh * 1.0f * IH / OH));
            int iw = (int) (Math.round(ow * 1.0f * IW / OW));
            for(int c=0; c<C; c++) {
                int yindex = (oh*OW + ow)*C + c;
                int xindex = (ih*IW + iw)*C + c;
                pixel2[yindex] = pixel[xindex];
            }
        }
        return pixel2;
    }
    
    
    public static void test1() throws IOException {
        BufferedImage img = cv.imread(path);
        int IH = img.getHeight();
        int IW = img.getWidth();
        byte[] pixel = cv.pixel(img);
        
        int OH1 = 100, OW1 = 100;
        byte[] pixel1 = resize(pixel, IH, IW, 3, OH1, OW1);
        img = cv.BGR(pixel1, OH1, OW1); cv.imshow(img, "2");
        System.out.println(cv.brief(img));
        
        int OH2 = IH + 100, OW2 = IW + 100;
        byte[] pixel2 = resize(pixel, IH, IW, 3, OH2, OW2);
        img = cv.BGR(pixel2, OH2, OW2); cv.imshow(img, "3");
        System.out.println(cv.brief(img));
    }
    
    public static void test2() throws IOException {
        BufferedImage img = cv.imread(path);
        int IH = img.getHeight(), IW = img.getWidth();
        Tensor X = eg.img.pixels(img);
        
        int OH1 = 100, OW1 = 100;
        img = cv.BGR(eg.img.resize(false, X, OH1, OW1)); cv.imshow(img, "x2");
        System.out.println(cv.brief(img));
        
        int OH2 = IH + 100, OW2 = IW + 100;
        img = cv.BGR(eg.img.resize(false, X, OH2, OW2)); cv.imshow(img, "x3");
        System.out.println(cv.brief(img));
    }
    
    public static void test3() throws IOException {
        BufferedImage img = cv.imread(path);
        int IH = img.getHeight(), IW = img.getWidth();
        Tensor X = eg.img.pixels(img);
        
        img = cv.BGR(eg.img.pad(true, X, 30, 30, 0)); cv.imshow(img, "x2");
        System.out.println(cv.brief(img));
        
        img = cv.BGR(eg.img.trim(true, X, 30, 30, 0)); cv.imshow(img, "x3");
        System.out.println(cv.brief(img));
    }
    
     public static void test3a() throws IOException {
        BufferedImage img = cv.imread(path);
        int IH = img.getHeight(), IW = img.getWidth();
        Tensor X = eg.img.pixels(img);
        
        BufferedImage img2 = cv.pad(img, 30, 30); cv.imshow(img2, "x2");
        System.out.println(cv.brief(img2));
        
        BufferedImage img3 = cv.trim(img, 30, 30); cv.imshow(img3, "x3");
        System.out.println(cv.brief(img3));
    }
    
    public static void test4() throws IOException {
        BufferedImage img = cv.imread(path);
        Tensor X = eg.img.pixels(img);
        
        SimpleTimer timer = SimpleTimer.clock();
        for(int i=0; i< 100; i++) {//370
            eg.img.pad(true, X, 30, 30, 0);
            eg.img.trim(true, X, 30, 30, 0);
            
            img = cv.BGR(eg.img.pad(true, X, 30, 30, 0)); 
            img = cv.BGR(eg.img.trim(true, X, 30, 30, 0));
        }
        long dif = timer.record().timeStamp_dif_millis();
        System.out.println(dif);
    }
    
    public static void test5() throws IOException {
        BufferedImage img = cv.imread(path);
        Tensor X = eg.img.pixels(img);
        
        SimpleTimer timer = SimpleTimer.clock();
        for(int i=0; i<1000; i++) {//1993
            Tensor X1 = eg.img.pad(false, X, 30, 30, 0); eg.img.pad(true, X1, 30, 30, 0);
            Tensor X2 = eg.img.trim(false, X, 30, 30, 0); eg.img.trim(true, X2, 30, 30, 0);
            img = cv.BGR(X1); X1.delete();
            img = cv.BGR(X2); X2.delete();
           
        }
        long dif = timer.record().timeStamp_dif_millis();
        System.out.println(dif);
    }
    
    public static void test5a() throws IOException {
        BufferedImage img = cv.imread(path);
        Tensor X = eg.img.pixels(img);
        
        SimpleTimer timer = SimpleTimer.clock();
        for(int i=0; i< 1000; i++) {//2875
            BufferedImage img2 = cv.pad(img, 30, 30);  img2 = cv.pad(img2, 30, 30); 
            BufferedImage img3 = cv.trim(img, 30, 30); img3 = cv.trim(img3, 30, 30);
        }
        long dif = timer.record().timeStamp_dif_millis();
        System.out.println(dif);
    }
    
    public static void test6() throws IOException {
        BufferedImage img = cv.imread(path);
        System.out.println(cv.brief(img));
        int IH = img.getHeight(), IW = img.getWidth();
        Tensor X = eg.img.pixels(img);
        
        img = cv.BGR(eg.img.expand(true, X, IH + 30, IW + 30, 3)); cv.imshow(img, "x2");
        System.out.println(cv.brief(img));
        
        img = cv.BGR(eg.img.crop(true, X, IH - 30, IW - 30, 3)); cv.imshow(img, "x3");
        System.out.println(cv.brief(img));
    }
    
    public static void main(String[] args)
    {
        try 
        {
//            test1();
            test2();
//            test3();
//            test4();
//            test3a();
            
            //test5();
//            test5a();
//            test6();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}
