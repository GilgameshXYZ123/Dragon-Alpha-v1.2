/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.dual.math;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.dual.DualCore;
import z.dragon.nn.unit.dual.DualFunction;

/**
 *
 * @author Gilgamesh
 */
public class Quadratic2Center extends DualFunction {
    private static final long serialVersionUID = 1L;
    
    protected int dim2;
    protected float k11, k12, k22, k1, k2, C;   
    
    public Quadratic2Center(int dim2,
            float k11, float k12, float k22,
            float k1, float k2,
            float C) 
    {
        this.dim2 = dim2;
        this.k11 = k11; this.k12 = k12; this.k22 = k22;
        this.k1  = k1; this.k2  = k2; 
        this.C = C;
    }

    //<editor-fold defaultstate="collapsed" desc="functions">
    public int dim2() { return dim2; }
    public Quadratic2Center dim2(int dim2) { this.dim2 = dim2; return this; }
             
    public float k11() { return k11; }
    public Quadratic2Center k11(float k11) { this.k11 = k11; return this; }
    
    public float k12() { return k12; }
    public Quadratic2Center k12(float k12) { this.k12 = k12; return this;}
    
    public float k22() { return k22; }
    public Quadratic2Center k22(float k22) { this.k22 = k22; return this;}
    
    public float k1() { return k1; }
    public Quadratic2Center k1(float k1) { this.k1 = k1; return this;}
    
    public float k2() { return k2; }
    public Quadratic2Center k2(float k2) { this.k2 = k2; return this; }
    
    public float C() { return C; }
    public Quadratic2Center C(float C) { this.C = C; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append("{ dim2 = ").append(dim2);
        sb.append(", [k11, k12, k22] = [")
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
        return new InlineQuadratic2Center(this);
    }
    //</editor-fold>
   
    //<editor-fold defaultstate="collapsed" desc="static class: InlineQuadratic2">
    public static class InlineQuadratic2Center extends DualCore<Quadratic2Center>  {
        public InlineQuadratic2Center(Quadratic2Center unit) { super(unit); }

        public int dim2() { return ut.dim2; }
        public float k11() { return ut.k11; }
        public float k12() { return ut.k12; }
        public float k22() { return ut.k22; }
        public float k1() { return ut.k1; }
        public float k2() { return ut.k2; }
        public float C() { return ut.C; }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X1, Tensor X2) {
            return eg.quadratic2_center(false, X1, X2, ut.dim2, ut.k11, ut.k12, ut.k22, ut.k1, ut.k2, ut.C);
        }

        @Override
        protected Tensor[] __backward__(Engine eg, Tensor deltaY,
                boolean grad_inplace, boolean backward_grads,
                boolean backward_grads1, boolean backward_grads2) 
        {
            if(!backward_grads) return null;
            
            Tensor X1 = holdX1(), X2 = holdX2();
            if(backward_grads1 && backward_grads2)//(1) deltaX1 = deltaY * (k11*2*X1 + k12*X2 + k1)
                return eg.quadratic2_center_deltaX(grad_inplace, deltaY, X1, X2, ut.dim2, ut.k11, ut.k12, ut.k22, ut.k1, ut.k2, ut.C);
            
            return new Tensor[] {//(2) deltaX2 = deltaY * (k22*2*X2 + k12*X1 + k2)
                (backward_grads1 ? eg.quadratic2_center_deltaX1(false, deltaY, X1, X2, ut.dim2, ut.k11, ut.k12, ut.k1) : null),
                (backward_grads2 ? eg.quadratic2_center_deltaX2(deltaY, X1, X2, ut.dim2, ut.k22, ut.k12, ut.k2) : null)
            };
        }
        //</editor-fold>
    }
    //</editor-fold>
}
