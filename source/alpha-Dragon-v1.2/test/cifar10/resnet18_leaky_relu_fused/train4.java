/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.resnet18_leaky_relu_fused;

import cifar10.resnet18_leaky_relu_fused.Net.ResNet18;
import java.awt.image.BufferedImage;
import java.io.File;
import static z.dragon.alpha.Alpha.alpha;
import static z.dragon.common.DragonCV.cv;
import z.dragon.data.BufferedTensorIter;
import z.dragon.data.Pair;
import z.dragon.data.ImageFolder;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;
import z.dragon.nn.loss.LossFunction;
import z.dragon.nn.optim.Optimizer;
import z.util.lang.SimpleTimer;

/**
 *
 * @author Gilgamesh
 */
public class train4 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 128);
    
    static String root_dir = "C:\\Users\\Gilgamesh\\Desktop\\check";
    
    static ImageFolder ds = alpha.data.image_folder(root_dir).init()
            .pixel_transform((File f)-> {  
                BufferedImage img = cv.imread(f);
                if(cv.channels(img) != 3) img = cv.to_BGR(img);
                img = cv.reshape(img, 128, 128);
                return cv.pixel(img);
            })
            .input_transform(alpha.data.pixel_to_tensor(128, 128, 3))
            .label_transform(alpha.data.onehot(10));
    
    static int batch_size = 128;//512;
    static float lr = 0.001f;//learning_rate
    
    public static void check_random_sort() {
        ds.fileFolder().shuffle_sort();
    }
    
    public static void training(int epoch) {
        ResNet18 net = new ResNet18().init(eg).println(); //net.load();
        net.train();
        
        Optimizer opt = alpha.optim.Adam(net.params(), lr);
//        Optimizer opt = alpha.optim.SGD(net.params(), lr).println();
        
        LossFunction loss = alpha.loss.softmax_crossEntropy();
        eg.sync(false).check(false);
        
        int batchIdx = 0; 
        SimpleTimer timer = new SimpleTimer().record();
        BufferedTensorIter iter = ds.buffered_iter(eg, batch_size);
        
        for(int i=0; i<epoch; i++) { 
            System.out.println("\n\nepoch = " + i);
            for(iter.reset().shuffle_sort(); iter.hasNext(); batchIdx++) {
                Pair<Tensor, Tensor> pair = iter.next();//iter.next(batch_size);
                Tensor x = pair.input.c().linear(true, 2.0f, -1.0f);
                Tensor y = pair.label;
                
                Tensor yh = net.forward(x)[0];
                
                if(batchIdx % 10 == 0) {
                    float ls = loss.loss(yh, y).get();
                    System.out.println(batchIdx + ": loss = " + ls + ", lr = " + opt.learning_rate());
                }
                
                net.backward(loss.gradient(yh, y));
                opt.update().clear_grads();
                net.gc();
            }
        }
        
        long div = timer.record().timeStamp_dif_millis();//3.7 - 11.187 = 7.48
        float time = 1.0f * div / epoch;
        System.out.println("total = " + (1.0f*div/1000));
        System.out.println("for each sample:" + time / batch_size);
        net.save();
    }
    
    public static void main(String[] args)
    {
        try
        {
            check_random_sort();
            //224*224*3: total = 35.466 -> 34.835 -> 33.603 ->  33.397 ->  33.179
            //100 batch, 128*128*128*3: 22.486, 22.841
            //25 epochs for Adam
            //50 epochs for SGD
            //30 epcohs for SGDMN
//            training(100);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}
