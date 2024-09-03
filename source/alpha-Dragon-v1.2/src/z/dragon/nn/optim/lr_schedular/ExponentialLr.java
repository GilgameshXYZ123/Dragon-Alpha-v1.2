/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.optim.lr_schedular;


/**
 *
 * @author Gilgamesh
 */
public class ExponentialLr extends LrSchedular {
    protected float gamma;
    protected float minLr;
    
    public ExponentialLr(float gamma, float minLr) {
        this.gamma(gamma);
        this.minLr = minLr;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float gamma() { return gamma; }
    public ExponentialLr gamma(float lamba) { 
        if(gamma <= 0.0f || gamma > 1.0f) throw new IllegalArgumentException(String.format(
                "gamma { got %f } must belongs to (0, 1)", gamma));
        this.gamma = lamba; 
        return this;
    }
    
    public float min_learning_rate() { return gamma; }
    public ExponentialLr min_learning_rate(float minLr) { this.minLr = minLr; return this; }
    
    @Override
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append(" {");
        sb.append("init_learning_rate = ").append(initLr);
        sb.append(", learning_rate = ").append(lr);
        sb.append(", gamma = ").append(gamma);
        sb.append(", min_learning_rate = ").append(minLr);
        sb.append(" }");
    }
    //</editor-fold>

    @Override
    public float next_learning_rate() {
        return lr = Math.max(minLr, lr * gamma);
    }
}
