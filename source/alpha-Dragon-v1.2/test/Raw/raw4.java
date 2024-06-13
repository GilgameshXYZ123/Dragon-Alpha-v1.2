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
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;

/**
 *
 * @author Gilgamesh
 */
public class raw4 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static String img_path  = "G:\\pepper-raw\\A.raw";
    static String white_path = "G:\\pepper-raw\\white.raw";
    static String dark_path = "G:\\pepper-raw\\dark.raw";
    
    public static void test2() throws IOException  {
        File file = new File(img_path);
        System.out.println(file.exists());
        System.out.println(file.length());
          
        int IH = 640, IW = 696, IC = 224;
        
        Tensor X = eg.img.read_raw_bil_dtype12(img_path, IH, IW, IC);
        Tensor rgb1 = X.img().extract_3channels(false, 13, 80, 132);
        BufferedImage img1 = cv.RGB(rgb1); cv.imshow(img1, "img1");
        
        Tensor white = eg.img.read_raw_bil_dtype12(white_path, 640, 694, IC)
                .img().resize(true, IH, IW);
        Tensor dark = eg.img.read_raw_bil_dtype12(dark_path, 640, 694, IC)
                .img().resize(true, IH, IW);
        System.out.println(X);
        
        Tensor rgb2 = white.img().extract_3channels(false, 13, 80, 132);
        Tensor rgb3 = dark.img().extract_3channels(false, 13, 80, 132);
        
        BufferedImage img2 = cv.RGB(rgb2); cv.imshow(img2, "img2");
        BufferedImage img3 = cv.RGB(rgb3); cv.imshow(img3, "img3");
        
        Tensor X2 = eg.img.reflection_normalization(true, X, white, dark)
                .img().linear_dtype_to_pixel(true, 255, 0);
        
        float brightness = 0.1f, saturation = 0.2f, contrast = 0.3f;
        Tensor rgb4 = X2
                .img().extract_3channels(false, 13, 80, 132)
                .img().adjust_color(true, brightness, saturation, contrast);
        BufferedImage img4 = cv.RGB(rgb4); cv.imshow(img4, "img4");
        
        System.gc();
    }
    
    public static void main(String[] args)
    {
        try
        {
            test2();
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }
}
