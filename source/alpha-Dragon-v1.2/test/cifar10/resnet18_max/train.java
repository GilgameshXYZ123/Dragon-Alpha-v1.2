/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.resnet18_max;

import static z.dragon.alpha.Alpha.alpha;
import cifar10.resnet18_max.Net.ResNet18;
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
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 128);
    static {
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
//        cu32.field_var_safe(false);
    }
    
    static int batch_size = 512;//512;
    static float lr = 0.001f;//learning_rate
    
    public static void training(int epoch) {
        ResNet18 net = new ResNet18().train().init(eg).println(); 
        
//        Optimizer opt = alpha.optim.Adam(net.param_map(), lr).amsgrad(true).L2(1e-2f);//25 epoch, 141.5s, 0.7807, 0.9870
//        Optimizer opt = alpha.optim.Adam(net.param_map(), lr).L2(1e-2f);//40 epoch, 229.ms, 0.793, 0.906

//        Optimizer opt = alpha.optim.Adamax(net.param_map(), lr);//25 epoch, 143.763, 0.7454, 0.989
//        Optimizer opt = alpha.optim.Adamax(net.param_map(), lr).L2(1e-2f);//50 epoch, 288.047, 0.7624, 0.9671
        
//        Optimizer opt = alpha.optim.Adamod(net.param_map(), lr);//35 epoch, 202.767, 0.7171, 0.981
//        Optimizer opt = alpha.optim.Adamod(net.param_map(), lr).L2(1e-2f);//50 epoch, 288.964, 0.7964, 0.905

//        Optimizer opt = alpha.optim.RAdam(net.params());//30 epoch, 174.419,  0.7396, 0.9849
//        Optimizer opt = alpha.optim.RAdam(net.params()).L2(1e-2f);//60 epoch, 345.753, 0.7619, 0.8874
        
        Optimizer opt = alpha.optim.AdamW(net.params()).amsgrad(true).println();//40 epoch, 230.219, 0.7887, 0.9955

//        Optimizer opt = alpha.optim.RMSprop(net.param_map());//30 epoch, 170.67, 0.751, 0.9909
//        Optimizer opt = alpha.optim.RMSprop(net.param_map()).learning_rate(0.001f).L2(1e-4f);//60 epoch, 343.841, 0.7317, 0.8999

//        Optimizer opt = alpha.optim.SGDMN(net.params(), lr).momentum(0.9f).nestorv(1);//35 epoch, 199.985, 0.6093, 1.0 
//        Optimizer opt = alpha.optim.SGDMN(net.params(), lr).momentum(0.9f).nestorv(1).L2(1e-2f);//50 eooch, 286.068, 0.5828, 1.0f

//        Optimizer opt = alpha.optim.SGD(net.params(), lr).println();
//        alpha.stat.load_zip(opt, opt_weight); 

        LossFunction loss = alpha.loss.softmax_crossEntropy();
        DataSet<byte[], Integer> train_set = Cifar10.train();
        BufferedTensorIter iter = train_set.buffered_iter(eg, batch_size);
        eg.sync(false).check(false);
        
        int batchIdx = 0; 
        SimpleTimer timer = new SimpleTimer().record();
        for(int i=0; i<epoch; i++) { 
            System.out.println("\n\nepoch = " + i);
            for(iter.shuffle_swap().reset(); iter.hasNext(); batchIdx++) {
                Pair<Tensor, Tensor> pair = iter.next();//iter.next(batch_size);
                Tensor X = pair.input.c().linear(true, 2.0f, -1.0f);
                Tensor Y = pair.label;
                
                Tensor yh = net.forward(X)[0];
                
                if(batchIdx % 10 == 0) {
                    float ls = loss.loss(yh, Y).get();
                    System.out.println(batchIdx + ": loss = " + ls + ", lr = " + opt.learning_rate());
                }
                
                net.backward(loss.gradient(yh, Y));
                opt.update().clear_grads();
                net.gc();
            }
        }
        
        long div = timer.record().timeStamp_dif_millis();//3.7 - 11.187 = 7.48
        float time = 1.0f * div / batchIdx;
        System.out.println("total = " + (1.0f*div/1000));
        System.out.println("for each sample:" + time/batch_size);
        net.save();
//        alpha.stat.save_zip(opt, opt_weight);
    }
    
    public static void main(String[] args)
    {
        try
        {
            //30.42: 30.332 -> 30.245 -> 29.122 -> 28.802 -> 28.203
            //25 epochs for Adam: 143.273
            //50 epochs for SGD
            //30 epcohs for SGDMN
            training(35);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}
