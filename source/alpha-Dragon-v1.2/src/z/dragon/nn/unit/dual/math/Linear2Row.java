/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.dual.math;

import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.dual.DualCore;
import z.dragon.nn.unit.dual.DualFunction;

/**
 *
 * @author Gilgamesh
 */
public class Linear2Row extends DualFunction {
    private static final long serialVersionUID = 1L;
    
    protected float alpha, beta, gamma;

    public Linear2Row(float alpha, float beta, float gamma) {
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions"> 
    public float alpha() { return alpha; }
    public Linear2Row alpha(float alpha) { this.alpha = alpha; return this; }
    
    public float beta() { return beta; }
    public Linear2Row beta(float beta) { this.beta = beta; return this; }
    
    public float gamma() { return gamma; }
    public Linear2Row gamma(float gamma) { this.gamma = gamma; return this; }

    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append("{ alpha = ").append(alpha);
        sb.append(", beta = ").append(beta);
        sb.append(", gamma = ").append(gamma).append(" }");
    }
    
    @Override
    protected InlineLinear2Row create_unit_core() {
        return new InlineLinear2Row(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineLinear2Row">
    public static class InlineLinear2Row extends DualCore<Linear2Row>
    {
        transient protected int row_length;
        
        public InlineLinear2Row(Linear2Row unit) { super(unit); }

        public float alpha() { return ut.alpha; }
        public float beta() { return ut.beta; }
        public float gamma() { return ut.gamma; }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X1, Tensor X2) {
            row_length = X2.length();
            return eg.linear2_row(false, X1, X2, ut.alpha, ut.beta, ut.gamma);
        }

        @Override
        protected Tensor[] __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads, 
                boolean backward_grads1, boolean backward_grads2) 
        {
            if(!backward_grads) return null;

            int gc_count = 0; Tensor deltaX1 = null, deltaX2 = null;
            if(backward_grads1) { 
                deltaX1 = eg.linear(false, ut.alpha, deltaY, 0);
                gc_count++;//(1) deltaX1 = deltaY * alpha
            }
            if(backward_grads) {
                deltaX2 = eg.field_linear(deltaY, row_length, ut.beta, 0);
                gc_count++;//(2) deltaX2 = field_sum: deltaY * beta
            }
            
            if(grad_inplace) {//when deltaX1 and deltaX2 are cauculated, deltaY is not needed
                CountGc gc = new CountGc(gc_count, deltaY);
                if(deltaX1 != null) deltaX1.dual(()-> { gc.countDown(); });
                if(deltaX2 != null) deltaX2.dual(()-> { gc.countDown(); });
            }
            return new Tensor[]{ deltaX1, deltaX2 };
        }
        //</editor-fold>
    }
    //</editor-fold>
}
