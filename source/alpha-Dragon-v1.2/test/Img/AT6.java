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

/**
 *
 * @author Gilgamesh
 */
public class AT6 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static String path = "C:\\Users\\Gilgamesh\\Desktop\\sj22.jpg";
    
    public static void test1() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img);
        int IH = img.getHeight(), IW = img.getWidth();
        
        Tensor X = eg.img.pixels(img);
        
        img = cv.BGR(eg.img.blacks_like(X)); cv.imshow(img, "x2");
        System.out.println(cv.brief(img));
        
        img = cv.BGR(eg.img.whites_like(X)); cv.imshow(img, "x3");
        System.out.println(cv.brief(img));
    }

    public static void test2() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img);
        
        Tensor X = eg.img.pixels(img);
        Tensor white = eg.img.constants_like(255, X);
        Tensor black = eg.img.constants_like(94, X);
        Tensor Y = eg.img.reflection_normalize(false, X, white, black)
                .tensor_to_pixel(true);
        
        BufferedImage img2 = cv.BGR(Y); cv.imshow(img2, "x2");
    }
   
    
    public static void test3() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img);
        System.out.println(cv.brief(img));
        int IH = img.getHeight(), IW = img.getWidth();
        
        Tensor X = eg.img.pixels(img);
        X = eg.img.pixel_to_dtype(true, X);
        
        eg.check(false);
        
        Tensor Y = eg.row_mean(X);
        Y = eg.img.dtype_to_pixel(true, Y);
        
        BufferedImage img2 = cv.gray(Y.img().pixel(), IH, IW); cv.imshow(img2, "x2");
    }
   
    public static BufferedImage absorb(BufferedImage img, float value) throws IOException {
        Tensor X = eg.img.pixels(img)
                .img().pixel_to_dtype(true);
        
        Tensor gray = eg.row_mean(X);
        
        X = eg.linear2_field(true, X, gray, (1 + value), -value, 0)
                .img().dtype_to_pixel(true);
        
        img = cv.BGR(X);
        
        gray.delete(); X.delete();
        return img;
    }
    
    public static void test4() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img);
        System.out.println(cv.brief(img));
        img = absorb(img, 0.5f); cv.imshow(img, "x2");
    }
    
    public static void test5() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img);
        System.out.println(cv.brief(img));
        
        Tensor X = eg.img.pixels(img);
//        X = eg.img.adjust_saturation(true, X, 1.4f);
//        X = eg.img.adjust_brightness(true, X, -1);
        X = eg.img.adjust_constrast(true, X, 0.2f);
        img = cv.BGR(X); cv.imshow(img, "x2");
        X.delete();
    }
    
    public static void main(String[] args)
    {
        try 
        {
            test5();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}
