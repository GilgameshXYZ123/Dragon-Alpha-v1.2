/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.resnet_lkn;

import cifar10.resnet_lkn.Net.ResLKN18;
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
public class train  {
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    static {
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
        cu32.tf32(true);
    }
    
    static int batch_size = 512;//512;
    static float lr = 0.0003f;//learning_rate
    
    public static void training(int epoch) {
        ResLKN18 net = new ResLKN18().train().init(eg).println(); //net.load();
       
//        Optimizer opt = alpha.optim.Adam(net.param_map(), lr).amsgrad(true);//25 epoch, 141.5s, 0.7807, 0.9870
//        Optimizer opt = alpha.optim.Adam(net.param_map(), lr).L2coef(1e-2f);//40 epoch, 229.ms, 0.793, 0.906

//        Optimizer opt = alpha.optim.Adamax(net.param_map(), lr);//25 epoch, 143.763, 0.7454, 0.989
//        Optimizer opt = alpha.optim.Adamax(net.param_map(), lr).L2coef(1e-2f);//50 epoch, 288.047, 0.7624, 0.9671
        
//        Optimizer opt = alpha.optim.Adamod(net.param_map(), lr);//35 epoch, 202.767, 0.7171, 0.981
//        Optimizer opt = alpha.optim.Adamod(net.param_map(), lr).L2coef(1e-2f);//50 epoch, 288.964, 0.7964, 0.905

//        Optimizer opt = alpha.optim.RAdam(net.params());//30 epoch, 174.419,  0.7396, 0.9849
//        Optimizer opt = alpha.optim.RAdam(net.params()).L2coef(1e-2f);//60 epoch, 345.753, 0.7619, 0.8874
        
        Optimizer opt = alpha.optim.AdamW(net.param_map(), lr, 1e-2f)
                .amsgrad(true)
                .lrSchedular(alpha.optim.cosAnnealingLr(10))
                .println();//40 epoch, 230.219, 0.7887, 0.9955
        
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
                
                Tensor grad = loss.gradient(yh, y);
                
                net.backward(grad);
                opt.update().clear_grads();
                net.gc();
            }
        }
        
        long div = timer.record().timeStamp_dif_millis();//3.7 - 11.187 = 7.48
        float time = 1.0f * div / batchIdx;
        System.out.println("total = " + (1.0f*div/1000));
        System.out.println("for each sample:" + time/batch_size);
        net.save(opt);
    }
    
    public static void main(String[] args) {
        try {
            training(0);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}