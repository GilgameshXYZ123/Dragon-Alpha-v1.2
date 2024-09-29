/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Img;

import static z.dragon.alpha.Alpha.alpha;
import static z.dragon.common.DragonCV.cv;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;

/**
 *
 * @author Gilgamesh
 */
public class test1 {
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2"); }
    static Mempool memp = alpha.engine.memp2(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static String path = "C:\\Users\\Gilgamesh\\Desktop\\alter.jpeg";
    
    public static void main(String[] args) {
        Tensor X = eg.img.pixels(cv.imread(path));
        int IH = X.dim(0), IW = X.dim(1);
        X = eg.img.random_affine()
                .random_rotate(0.1f)
                .random_shear(0.1f)
                .random_translate(0.1f, IH, IW)
                .random_vertical_filp(0.5f, IH)
                .random_horizontal_flip(0.5f, IW)
                .transform(X, true, IH, IW);
        
        cv.imshow(eg.img.BGR(X));
        
        
    }
}
