/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Img;

import static Img.AT6.path;
import static Img.AT6.test5;
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
public class AT7 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static String path = "C:\\Users\\Gilgamesh\\Desktop\\sj22.jpg";

    public static BufferedImage adjust_constrast1(BufferedImage img, float value) throws IOException {
        Tensor X = eg.img.pixels(img);
        Tensor fX = X.img().pixel_to_dtype(false);
        float xmax = fX.max().get();
        float xmin = fX.min().get();
        
        float ymax = 255, ymin = 0;
        float a = (ymax - ymin) / (xmax - xmin);
        float b = (xmax*ymin - ymax*xmin) / (xmax - xmin);
        
        System.out.println(a + ", " + b);
        
        X = eg.img.linear(true, a, X, b);
        img = cv.BGR(X); X.delete();
        return img;
    }
    
    public static BufferedImage adjust_constrast2(BufferedImage img, float value) throws IOException {
        Tensor X = eg.img.pixels(img);
        Tensor fX = X.img().pixel_to_dtype(false);
        float xmax = fX.max().get();
        float xmin = fX.min().get();
        
        float incr = (value - 1) / 2 + 1;
        float ymax = 255*incr, ymin = -255*incr + 255;
        float a = (ymax - ymin) / (xmax - xmin);
        float b = (xmax*ymin - ymax*xmin) / (xmax - xmin);
        
        System.out.println(a + ", " + b);
        
        X = eg.img.linear(true, a, X, b);
        img = cv.BGR(X); X.delete();
        return img;
    }
    
    public static BufferedImage adjust_constrast3(BufferedImage img, float value) throws IOException {
        Tensor X = eg.img.pixels(img);
        Tensor fX = X.img().pixel_to_dtype(false);
        float mean = fX.mean().get();
        
        float a = value;
        float b = (float) (mean * (1 - Math.sqrt(value)));
        
        X = eg.img.linear(true, a, X, b);
        img = cv.BGR(X); X.delete();
        return img;
    }
   
    public static void test1() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img);
        img = adjust_constrast1(img, 0); cv.imshow(img, "ssss");
    }
    
    public static void test2() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img);
        img = adjust_constrast2(img, 2f); cv.imshow(img, "ssss");
    }
    
    public static void test3() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img);
        img = adjust_constrast3(img, 0f); cv.imshow(img, "ssss");
    }

    public static void test4() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img);
        for(int i=0; i<5; i++) {
            Tensor X = eg.img.pixels(img);
            X = eg.img.random_expand(true, X, 1300, 1300, 3);
            BufferedImage img2 = cv.BGR(X); X.delete();
            cv.imshow(img2, "sss:" + i);
        }
    }
    
    public static void test5() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img);
        Tensor X = eg.img.pixels(img);
        X = eg.img.adjust_color(true, X, 0, 0.5f, 0.5f);
        BufferedImage img2 = cv.BGR(X);
        X.delete(); cv.imshow(img2, "sssv2:");
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
