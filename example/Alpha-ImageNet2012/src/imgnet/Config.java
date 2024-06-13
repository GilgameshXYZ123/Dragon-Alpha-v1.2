/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package imgnet;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.function.Function;
import static z.dragon.alpha.Alpha.alpha;
import static z.dragon.common.DragonCV.cv;
import z.dragon.data.ImageFolder;
import z.dragon.data.Transform;
import z.dragon.engine.Engine;
import z.dragon.engine.memp.Mempool;

/**
 *
 * @author Gilgamesh
 */
public class Config 
{
    public static final String alpha_home = "C:\\Users\\Tourist\\Desktop\\Dragon-alpha-v1.2";
    public static final String weight_home = "C:\\Users\\Tourist\\Desktop\\";
    
    public static final String imgnet2012_home = "D:\\virtual-disc-V-data\\Imagenet2012\\";
    public static final String imgnet2012_trainset     = imgnet2012_home + "ILSVRC2012_img_train";
    public static final String imgnet2012_valset       = imgnet2012_home + "ILSVRC2012_img_val";
    public static final String imgnet2012_testset      = imgnet2012_home + "ILSVRC2012_img_test";
    
    static { alpha.home(Config.alpha_home); }
    static final Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 22);
    public static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 2048);
    
    static final int img_size = 128;
    
    static final Function<File, byte[]> pixel_transform = (File f) ->{
        BufferedImage img = cv.imread(f);
        if(cv.channels(img) != 3) img = cv.to_BGR(img);
        return cv.pixel(cv.reshape(img, img_size, img_size));
    };
    
    static final Transform input_transform = alpha.data.pixel_to_tensor(img_size, img_size, 3);
    static final Transform label_transform = alpha.data.onehot(1000);
    
    public static ImageFolder ImgNet2012_trainset = alpha.data.image_folder(Config.imgnet2012_trainset)
            .pixel_transform(pixel_transform)
            .input_transform(input_transform)
            .label_transform(label_transform)
            .num_threads(16);
    
     public static ImageFolder ImgNet2012_valset = alpha.data.image_folder(Config.imgnet2012_valset)
            .pixel_transform(pixel_transform)
            .input_transform(input_transform)
            .label_transform(label_transform)
            .num_threads(16);
}
