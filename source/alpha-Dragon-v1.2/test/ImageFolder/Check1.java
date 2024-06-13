/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ImageFolder;

import java.awt.image.BufferedImage;
import java.io.File;
import static z.dragon.alpha.Alpha.alpha;
import static z.dragon.common.DragonCV.cv;
import z.dragon.data.ImageFolder;
import z.dragon.data.Pair;
import z.dragon.engine.Engine;
import z.dragon.engine.memp.Mempool;

/**
 *
 * @author Gilgamesh
 */
public class Check1 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 128);
    
    static String root_dir = "C:\\Users\\Gilgamesh\\Desktop\\check";
    
    static ImageFolder ds = alpha.data.image_folder(root_dir)
            .pixel_transform((File f)-> {  
                BufferedImage img = cv.imread(f);
                if(cv.channels(img) != 3) img = cv.to_BGR(img);
                img = cv.reshape(img, 128, 128);
                cv.imshow(img, f.toString());
                return cv.pixel(img);
            })
            .input_transform(alpha.data.pixel_to_tensor(128, 128, 3))
            .label_transform(alpha.data.onehot(10));
    
    public static void main(String[] args) {
        ds.init();
        System.out.println(ds.labels());
        ds.fileFolder().container().get(8);
        
        
    }
}
