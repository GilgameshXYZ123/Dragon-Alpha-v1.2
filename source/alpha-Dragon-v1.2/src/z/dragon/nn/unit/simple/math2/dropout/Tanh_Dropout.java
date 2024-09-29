/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math2.dropout;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;

/**
 * Tanh(dropout(X))
 * @author Gilgamesh
 */
public class Tanh_Dropout extends Dropout {
    private static final long serialVersionUID = 531781240350001L;
    
    public Tanh_Dropout(boolean inplace, float nonzero_p) {
        super(inplace, nonzero_p);
    }
    
     @Override
    protected InlineTanh_Dropout create_unit_core() {
        return new InlineTanh_Dropout(this);
    }
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineTanh_Dropout">
    public static class InlineTanh_Dropout extends InlineDropout<Tanh_Dropout> {
        public InlineTanh_Dropout(Tanh_Dropout unit) { super(unit); }
    
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            if(!ut.training) return eg.tanh(inplace, X);//exp = (1/p)*p + 0*(1-p) = 1

            Tensor[] outs = eg.tanh_dropout(X, ut.p);
            R = outs[1];
            return outs[0];
        }
        //</editor-fold>
    }
    //</editor-fold>
}
