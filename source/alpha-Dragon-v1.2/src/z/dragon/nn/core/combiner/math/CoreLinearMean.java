/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.combiner.math;

import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.combiner.CombinerCore;
import z.dragon.nn.unit.combiner.Combiner;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreLinearMean<T extends Combiner> extends CombinerCore<T> {
    protected float alpha;
    protected float beta;
    
    transient protected float mean_alpha;
    transient protected float mean_beta;

    public CoreLinearMean(T unit, float alpha, float beta) {
        super(unit);
        this.alpha = alpha;
        this.beta = beta;
    }
    
    public final float alpha() { return alpha; }
    public final float beta() { return beta; }

    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor[] X) {
        mean_alpha = (alpha / X.length);
        mean_beta  = (beta  / X.length);
        return eg.linear_sum(false, mean_alpha, mean_beta, X);
    }

     @Override
    protected Tensor[] __backward__(Engine eg, Tensor deltaY, 
            int input_tensor_num, boolean grad_inplace, 
            boolean backward_grads, boolean[] last_need_grads) 
    {
        if(!backward_grads) return null;//grads_num = 0;
        
        Tensor[] deltaX = new Tensor[input_tensor_num];
        int[] backward_grads_index = backward_grads_index(last_need_grads);
        int grads_num = backward_grads_index.length;
        
        if(grads_num == 1) {
            Tensor grad = eg.linear(grad_inplace, mean_alpha, deltaY, 0);
            int idx = backward_grads_index[0]; deltaX[idx] = grad;
            return deltaX;
        }
        if(grads_num == 2) {
            Tensor[] grads = eg.linear_2out(false, deltaY, mean_alpha, 0, mean_alpha, 0);
            int idx1 = backward_grads_index[0]; deltaX[idx1] = grads[0];
            int idx2 = backward_grads_index[1]; deltaX[idx2] = grads[1];
            return deltaX;
        }
        
        int grads_num2 = (grads_num >> 1 << 1), index = 0;
        while(index < grads_num2) {
            Tensor[] grads = eg.linear_2out(false, deltaY, mean_alpha, 0, mean_alpha, 0);
            int idx1 = backward_grads_index[index++]; deltaX[idx1] = grads[0];
            int idx2 = backward_grads_index[index++]; deltaX[idx2] = grads[1];
        }
        if(index < grads_num) {//remainder: grads_num % 2 != 0
            Tensor grad = eg.linear(grad_inplace, mean_alpha, deltaY, 0);
            int idx = backward_grads_index[index]; deltaX[idx] = grad;
            return deltaX;
        }
        
        if(grad_inplace) {//when deltaX[] are found, deltaY is not needed
            CountGc gc = new CountGc(grads_num, deltaY);
            for (Tensor grad : deltaX) grad.dual(()-> { gc.countDown(); });
        }
        
        return deltaX;
    }
    //</editor-fold>
}
