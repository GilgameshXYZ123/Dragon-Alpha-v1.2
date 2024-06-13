/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.resnet18_leaky_relu;

import cifar10.resnet18_leaky_relu.Net.ResNet18;
import java.io.IOException;
import static z.dragon.alpha.Alpha.alpha;
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
public class train3 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static int batch_size = 512;//512;
    static float lr = 0.001f;//learning_rate
    
    public static void training(int nIter) throws IOException
    {
        ResNet18 net = new ResNet18().init(eg).println(); //net.load();
        net.train();
        
        Optimizer opt = alpha.optim.Adam(net.params(), lr);
//        Optimizer opt = alpha.optim.SGD(net.params(), lr).println();
        
        LossFunction loss = alpha.loss.softmax_crossEntropy();
        eg.sync(false).check(false);
      
        SimpleTimer timer = new SimpleTimer().record();
        for (int i = 0; i < nIter; i++) {
//            Tensor x = eg.zeros(batch_size, 128, 128, 3);
            Tensor x = eg.zeros(batch_size, 32, 32, 3);
            Tensor y = eg.zeros(batch_size, 10);

            Tensor yh = net.forward(x)[0];

            if (i % 10 == 0) {
                float ls = loss.loss(yh, y).get();
                System.out.println(i + ": loss = " + ls + ", lr = " + opt.learning_rate());
            }

            net.backward(loss.gradient(yh, y));
            
            opt.update().clear_grads();
            net.gc();
            System.gc();
        }
        
        long div = timer.record().timeStamp_dif_millis();//3.7 - 11.187 = 7.48
        float time = 1.0f * div / nIter;
        System.out.println("total = " + (1.0f*div/1000));
        System.out.println("for each sample:" + time / batch_size);
        net.save();
    }
    
    public static void main(String[] args)
    {
        try
        {
            //1.1: 
            //1.05:
            
            //50: 30.69
            
            //224*224*3: total = 35.121
            //100 batch, 128*128*128*3: 22.486, 22.841
            //25 epochs for Adam
            //50 epochs for SGD
            //30 epcohs for SGDMN
            training(100000);
        }
        catch(IOException e) {
            e.printStackTrace();
        }
    }
}
