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
public class CosAnnealingLr extends LrSchedular {
    protected float minLr;
    protected float tmax;
    protected int epoch;
    protected float threshold;
    protected float period;
    
    public CosAnnealingLr(float tmax, float min_lr) {
        this.tmax(tmax);
        this.minLr = min_lr;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float min_learning_rate() { return minLr; }
    public CosAnnealingLr min_learning_rate(float minLr) { this.minLr = minLr; return this; }
    
    public float tmax() { return tmax; }
    public CosAnnealingLr tmax(float tmax) {
        if(tmax == 0) throw new IllegalArgumentException("tmax can't be zero."); 
        this.tmax = tmax; 
        return this; 
    }
    
    @Override
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append(" {");
        sb.append("init_learning_rate = ").append(initLr);
        sb.append(", learning_rate = ").append(lr);
        sb.append(", min_learning_rate = ").append(minLr);
        sb.append(", tmax = ").append(tmax);
        sb.append(" }");
    }
    //</editor-fold>

    @Override
    public void init(float initLr) {
        super.init(initLr); 
        this.epoch = 0;
        this.threshold = initLr - minLr;
        this.period = (float) (Math.PI / tmax);
    }

    @Override
    public float next_learning_rate() {
        return lr = (float) (minLr + threshold * (1 + Math.cos(epoch++ * period)));
    }
}
