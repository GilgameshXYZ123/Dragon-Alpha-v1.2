/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine;

/**
 * 
 * @author Gilgamesh
 */
public abstract class Counter {
    private int count;
    private boolean runned = false;
    
    public Counter(int count) {
        if(count <= 0) throw new IllegalArgumentException(String.format("count { got %d } must > 0", count));
        this.count = count;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-functions">
    public final int count() { return count;}
    public final boolean runned() { return runned; }

    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append(" {");
        sb.append("count = ").append(count);
        sb.append(", runned = ").append(runned);
        sb.append(" }");
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(256);
        this.append(sb);
        return sb.toString();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    public abstract void run();
    public synchronized void countDown() {
        count--;
        if (count <= 0 && !runned) {
            run();
            runned = true;
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: CountGc">
    public static class CountGc extends Counter  
    { 
        private final Tensor ts;
        
        public CountGc(int count, Tensor ts) {
            super(count);
            this.ts = ts;
        }
        
        public final Tensor tensor() { return ts; }
        
        @Override 
        public final void run() { 
            if(ts != null) ts.delete(); 
        }
    }
    //</editor-fold>
    public static CountGc countGc(int count, Tensor ts) {
        return new CountGc(count, ts);
    }
}
