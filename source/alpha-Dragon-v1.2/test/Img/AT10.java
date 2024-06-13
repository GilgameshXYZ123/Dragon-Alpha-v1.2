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
public class AT10
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static String path = "C:\\Users\\Gilgamesh\\Desktop\\sj22.jpg";
    
    public static void test1() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img);
        
        Tensor X = eg.img.pixels(img); 
//        X = eg.img.crop2D(true, X, new int[]{ 512, 256 }, null);
//        X = eg.img.crop(true, X, new int[]{ 512, 256, 0 }, null);
        //new int[] { 162, 944, 3 }
        
        X = eg.img.pixel_to_dtype(true, X);
        X = eg.crop2D(true, X, new int[]{ 0, 164 }, null)
                .img().dtype_to_pixel(true);
        
        cv.imshow(cv.BGR(X));
    }
    
    public static void test2() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img);
        
        Tensor X = eg.img.pixels(img); 
        X = eg.img.expand2D(true, X, new int[]{ 512, 256 }, null);
        X = eg.img.crop2D(true, X, new int[] {512,  256}, null);
        
//        X = eg.img.crop(true, X, new int[]{ 512, 256, 0 }, null);
        //new int[] { 162, 944, 3 }
        
//        X = eg.img.pixel_to_dtype(true, X);
//        X = eg.expand2D(true, X, new int[]{ 12, 164 }, null);
//        X = eg.crop2D(true, X, new int[]{ 12, 164 }, null)
//                .img().dtype_to_pixel(true);
        
        cv.imshow(cv.BGR(X));
    }
    
    public static void test3() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img);
        
        Tensor X = eg.img.pixels(img); 
//        X = eg.img.pad2D(true, X, 12, 12);
//        X = eg.img.trim2D(true, X, 12, 12);
        
        X = eg.img.pixel_to_dtype(true, X);
        X = eg.pad2D(true, X, 12);
        X = eg.trim2D(true, X, 12);
//        X = eg.crop2D(true, X, new int[]{ 12, 164 }, null)
        X = eg.img.dtype_to_pixel(true, X);
        
        cv.imshow(cv.BGR(X));
    }
    
    public static void test4() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img);
        
        Tensor X = eg.img.pixels(img); 
        for(int i=0; i<4; i++) {
            Tensor Y = X.img().random_crop(false, 400, 300, 3);
            cv.imshow(cv.BGR(Y));
        }
    }
    
    public static void main(String[] args) {
        try 
        {
            test4();
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
}
