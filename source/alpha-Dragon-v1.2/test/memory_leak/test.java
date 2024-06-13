/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package memory_leak;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class test
{  
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);

    public static void create_test(int height, int width) {
        float[] X = Vector.random_float_vector(height * width);
        for(int i=0; i<100000; i++) {
            Tensor tX = eg.tensor(X, height, width);
            tX.delete();
        }
    }
    
    public static void create_test_int8_2_float(int height, int width) {
        byte[] X = Vector.random_byte_vector(height * width);
        for(int i=0; i<100000; i++) {
            Tensor tX = eg.tensor(X, height, width);
            tX.delete();
            System.gc();
        }
    }
    
    public static void create_test_pixel_2_tensor(int height, int width) {
        byte[] X = Vector.random_byte_vector(height * width);
        for(int i=0; i<100000; i++) {
            Tensor tX = eg.pixel_to_tensor(X, height, width);
            tX.delete();
            System.gc();
        }
    }
    
    public static void create_test_int8(int height, int width) {
        byte[] X = Vector.random_byte_vector(height * width);
        for(int i=0; i<100000; i++) {
            Tensor tX = eg.tensor_int8(X, height, width);
            tX.delete();
            System.gc();
        }
    }
    
    
    public static void main(String[] args)//1426
    {
        //create_test(512, 128*128*3);
        //create_test_int8_2_float(512, 224*224*3);
        //create_test_pixel_2_tensor(512, 224*224*3);
        create_test_int8(512, 224*224*3);
        
    }
    
}
