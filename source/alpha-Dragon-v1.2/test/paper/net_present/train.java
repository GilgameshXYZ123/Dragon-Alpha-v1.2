/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package paper.net_present;

import paper.net_present.Net.Network;
import static paper.net_present.Net.eg;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.data.BufferedTensorIter;
import z.dragon.data.Pair;
import z.dragon.dataset.Cifar10;
import z.dragon.engine.Tensor;
import z.dragon.nn.loss.LossFunction;
import z.dragon.nn.optim.Optimizer;

/**
 *
 * @author Gilgamesh
 */
public class train 
{
    static float lr = 0.001f;
    static int batchsize = 512;
    static int epoch = 5;
    
    public static void train() {
        Network net = new Network().train().init(eg).println();
        Optimizer opt = alpha.optim.Adam(net.param_map(), lr).println();
        LossFunction ls = alpha.loss.softmax_crossEntropy();
        BufferedTensorIter iter = Cifar10.train().buffered_iter(eg, batchsize);

        alpha.stat.load_zip(net, "weight");
        alpha.stat.load_zip(opt, "opt_weight"); 
        eg.sync(false).check(false);
        for(int i=0; i<epoch; i++) 
            for(iter.shuffle_sort().reset(); iter.hasNext(); ) {
                Pair<Tensor, Tensor> pair = iter.next();
                Tensor x = pair.input;
                Tensor y = pair.label;
                
                Tensor yh = net.forward(x)[0];
                alpha.print("loss = ", ls.loss(yh, y));
                net.backward(ls.gradient(yh, y));
                opt.update().clear_grads();
                net.gc();
            }
        alpha.stat.save_zip(net, "weight"); 
        alpha.stat.save_zip(opt, "opt_weight");
    }
}
