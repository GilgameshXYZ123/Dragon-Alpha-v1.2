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
import z.dragon.engine.ImageEngine.ImageAffiner;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;

/**
 *
 * @author Gilgamesh
 */
public class AT9 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static String path = "C:\\Users\\Gilgamesh\\Desktop\\sj22.jpg";
    
    public static void test1() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img);
        
        Tensor X = eg.img.pixels(img); 
        ImageAffiner af = eg.img.affine()
                .rotate(0.5f)
                .shear(0.2f, 0.2f)
                .rotate(0.5f);
        
        ImageAffiner fa = af.copy().invert();
        
        X = af.transform(X, true); cv.imshow(cv.BGR(X));
        X = fa.transform(X, true); cv.imshow(cv.BGR(X));
    }
    
    public static void main(String[] args) throws IOException
    {
        test1();
    }
}
