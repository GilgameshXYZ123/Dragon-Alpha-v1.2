/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.resnet18_leaky_relu;

import cifar10.resnet18_leaky_relu.Net.ResNet18;
import java.io.IOException;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.data.BufferedTensorIter;
import z.dragon.data.DataSet;
import z.dragon.data.Pair;
import z.dragon.dataset.Cifar10;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.dragon.engine.memp.Mempool;
import z.dragon.nn.loss.LossFunction;
import z.dragon.nn.optim.Optimizer;
import z.util.lang.SimpleTimer;

/**
 *
 * @author Gilgamesh
 */
public class train 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2"); }
    final static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
//    static Engine eg = alpha.engine.cuda_float32(0, 1, memp, alpha.MEM_1MB*1024, true);//30.167
    
    static {
        eg.kaiming_fan_mode = Engine.FanMode.fan_in_out;
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
//        cu32.conv3D_useTexture(true);
    }
    
    static int batch_size = 512;//512;
    static float lr = 0.001f;//learning_rate
    
    public static void training(int epoch) throws IOException
    {
        ResNet18 net = new ResNet18().init(eg).println(); //net.load();
        net.train();
      
        Optimizer opt = alpha.optim.Adam(net.params(), lr);
//        Optimizer opt = alpha.optim.SGDMN(net.params(), lr).momentum(0.9f).nestorv(1);
//        Optimizer opt = alpha.optim.SGD(net.params(), lr).println();
        
        LossFunction loss = alpha.loss.softmax_crossEntropy();
        DataSet<byte[], Integer> train_set = Cifar10.train();
        BufferedTensorIter iter = train_set.buffered_iter(eg, batch_size);
        eg.sync(false).check(false);
        
        int batchIdx = 0; 
        SimpleTimer timer = new SimpleTimer().record();
        for(int i=0; i<epoch; i++) { 
            System.out.println("\n\nepoch = " + i);
            for(iter.reset().shuffle_swap(); iter.hasNext(); batchIdx++) {
                Pair<Tensor, Tensor> pair = iter.next();
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
        float time = 1.0f * div / batchIdx;
        System.out.println("total = " + (1.0f*div/1000));
        System.out.println("for each sample:" + time/batch_size);
        net.save();
    }
    
    public static void main(String[] args)
    {
        try
        {
            //5 epochs: 30.451
            //25 epochs for Adam 
            //50 epochs for SGD
            //30 epcohs for SGDMN
            training(25);
        }
        catch(IOException e) {
            e.printStackTrace();
        }
    }
}
