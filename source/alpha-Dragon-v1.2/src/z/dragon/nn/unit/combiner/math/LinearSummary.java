/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.combiner.math;

import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.combiner.CombinerCore;
import z.dragon.nn.unit.combiner.CombinerFunction;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class LinearSummary extends CombinerFunction {
    private static final long serialVersionUID = 1L;
    
    protected float alpha;
    protected float beta;
    
    public LinearSummary(float alpha, float beta) {
        this.alpha = alpha;
        this.beta = beta;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public float alpha() { return alpha; }
    public LinearSummary alpha(float alpha) {this.alpha = alpha; return this;}
    
    public float beta() { return beta; }
    public LinearSummary beta(float beta) {this.beta = beta; return this;}
    
    @Override
    public void append(String pre, StringBuilder sb) { 
        sb.append(pre).append(default_name());
        sb.append("{ alpha = ").append(alpha);
        sb.append(", beta = ").append(beta).append(" }");
    }
    
    @Override
    protected InlineLinearSummary create_unit_core() {
        return new InlineLinearSummary(this);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="static class: InlineLinearSummary"> 
    public static class InlineLinearSummary extends CombinerCore<LinearSummary>
    {
        public InlineLinearSummary(LinearSummary unit) { super(unit); }

        public final float alpha() { return ut.alpha; }
        public final float beta() { return ut.beta; }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor[] X) {
            return eg.linear_sum(false, ut.alpha, ut.beta, X);
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
                Tensor grad = eg.linear(grad_inplace, ut.alpha, deltaY, 0);
                int idx = backward_grads_index[0]; deltaX[idx] = grad;
                return deltaX;
            }
            if(grads_num == 2) {
                Tensor[] grads = eg.linear_2out(false, deltaY, ut.alpha, 0, ut.alpha, 0);
                int idx1 = backward_grads_index[0]; deltaX[idx1] = grads[0];
                int idx2 = backward_grads_index[1]; deltaX[idx2] = grads[1];
                return deltaX;
            }
            
            int grads_num2 = (grads_num >> 1 << 1), index = 0;
            while(index < grads_num2) {
                Tensor[] grads = eg.linear_2out(false, deltaY, ut.alpha, 0, ut.alpha, 0);
                int idx1 = backward_grads_index[index++]; deltaX[idx1] = grads[0];
                int idx2 = backward_grads_index[index++]; deltaX[idx2] = grads[1];
            }
            if(index < grads_num) {//remainder: grads_num % 2 != 0
                Tensor grad = eg.linear(grad_inplace, ut.alpha, deltaY, 0);
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
    //</editor-fold>
}
