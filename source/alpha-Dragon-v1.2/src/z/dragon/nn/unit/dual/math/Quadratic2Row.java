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
public class Quadratic2Row extends DualFunction 
{
    private static final long serialVersionUID = 1L;
    
    protected float k11, k12, k22, k1, k2, C;   
    
    public Quadratic2Row( 
            float k11, float k12, float k22,
            float k1, float k2,
            float C) 
    {
        this.k11 = k11; this.k12 = k12; this.k22 = k22;
        this.k1  = k1; this.k2  = k2; 
        this.C = C;
    }
 
    //<editor-fold defaultstate="collapsed" desc="functions">
    public float k11() { return k11; }
    public Quadratic2Row k11(float k11) { this.k11 = k11; return this; }
    
    public float k12() { return k12; }
    public Quadratic2Row k12(float k12) { this.k12 = k12; return this;}
    
    public float k22() { return k22; }
    public Quadratic2Row k22(float k22) { this.k22 = k22; return this;}
    
    public float k1() { return k1; }
    public Quadratic2Row k1(float k1) { this.k1 = k1; return this;}
    
    public float k2() { return k2; }
    public Quadratic2Row k2(float k2) { this.k2 = k2; return this; }
    
    public float C() { return C; }
    public Quadratic2Row C(float C) { this.C = C; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append("{ [k11, k12, k22] = [")
                .append(k11).append(", ")
                .append(k12).append(", ")
                .append(k22).append(']');
        sb.append(", [k1, k2, C] = ")
                .append(k1).append(", ")
                .append(k2).append(", ")
                .append(C).append("] }");
    }
    
    @Override
    protected DualCore<?> create_unit_core() {
        return new InlineQuadratic2Row(this);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="static class: InlineQuadratic2">
    public static class InlineQuadratic2Row extends DualCore<Quadratic2Row> 
    {
        public InlineQuadratic2Row(Quadratic2Row unit) { super(unit); }

        public float k11() { return ut.k11; }
        public float k12() { return ut.k12; }
        public float k22() { return ut.k22; }
        public float k1() { return ut.k1; }
        public float k2() { return ut.k2; }
        public float C() { return ut.C; }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X1, Tensor X2) {
            return eg.quadratic2_row(false, X1, X2, ut.k11, ut.k12, ut.k22, ut.k1, ut.k2, ut.C);
        }

        @Override
        protected Tensor[] __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads, 
                boolean backward_grads1, boolean backward_grads2) 
        {
            if(!backward_grads) return null;
            
            int gc_count = 0; Tensor deltaX1 = null, deltaX2 = null;
            Tensor X1 = holdX1(), X2 = holdX2();
            
            if(backward_grads1) {
                deltaX1 = eg.quadratic2_row_deltaX1(false, deltaY, X1, X2, ut.k11, ut.k12, ut.k1);
                gc_count++;//(1) deltaX1 = deltaY * (k11*2*X1 + k12*X2 + k1)
            }
            if(backward_grads2) {//row_length = X2.length
                deltaX2 = eg.quadratic2_row_deltaX2(deltaY, X1, X2, X2.length(), ut.k22, ut.k12, ut.k2);
                gc_count++;//(2) deltaX2 = deltaY * (k22*2*X2 + k12*X1 + k2)
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
