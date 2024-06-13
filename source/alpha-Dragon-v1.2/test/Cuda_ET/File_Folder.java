/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Cuda_ET;

import java.awt.image.BufferedImage;
import java.io.File;
import static z.dragon.alpha.Alpha.alpha;
import static z.dragon.common.DragonCV.cv;
import z.dragon.data.DataSet;
import z.dragon.data.TensorIter.TensorPair;
import z.dragon.data.FileFolder;
import z.dragon.data.ImageFolder;
import z.dragon.engine.Engine;
import z.dragon.engine.memp.Mempool;

/**
 *
 * @author Gilgamesh
 */
public class File_Folder 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp3(alpha.MEM_1GB * 6);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    static String root_dir = "C:\\Users\\Gilgamesh\\Desktop\\check";
    
    public static void test1() {
        FileFolder ff = alpha.data.file_folder("C:\\Users\\Gilgamesh\\Desktop\\check").init();
        System.out.println(ff.container().get(100));
        System.out.println(ff);
    }
    
    public static void test2() {
        ImageFolder ds = alpha.data.image_folder(root_dir).init()
                .pixel_transform((File f)-> {  
                    BufferedImage img = cv.imread(f);
                    if(cv.channels(img) != 3) img = cv.to_BGR(img);
                    img = cv.reshape(img, 128, 128);
                    return cv.pixel(img);
                })
                .input_transform(alpha.data.int8_to_tensor(128, 128, 3))
                .label_transform(alpha.data.onehot(1000));
        
        TensorPair kv = ds.get(eg, 32);
        System.out.println(kv);
    }
    
    public static void main(String[] args)
    {
        test2();
    }
}
