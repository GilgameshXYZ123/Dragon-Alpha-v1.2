/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Img;

import static Img.AT7.path;
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
public class AT8 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static String path = "C:\\Users\\Gilgamesh\\Desktop\\sj22.jpg";
    
    public static void test1() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img);
        
        Tensor X = eg.img.pixels(img);
        System.out.println(X);
        int IH = img.getHeight(), IW = img.getWidth();
        float[] M = { -1, 0, IW - 1,
                       0, 1, 0 };
        
//        X = eg.img.affine(true, X, IH, IW, M);
//        X = eg.img.affine()
//                .vertical_flip(IH)
//                .horizontal_flip(IW)
//                .transform(X, true);

           
//        X = eg.img.vertical_flip(true, X, IH, IW);
//        X = eg.img.horizontal_flip(true, X, IH, IW);
        
        X = X.img().horizontal_flip(true, IH, IW);

        System.out.println(X);
        
        img = cv.BGR(X); cv.imshow(img, "ssss");
    }
    
    public static void main(String[] args)
    {
        try
        {
            test1();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}
