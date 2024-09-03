/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import z.dragon.data.TensorIter.TensorPair;
import z.dragon.engine.Engine;

/**
 *
 * @author Gilgamesh
 */
public class BufferedTensorIter {
    private static final ThreadFactory daemonThreadFactory = (Runnable r) -> {
        Thread t = new Thread(r);
        t.setDaemon(true);
        return t;
    };
    private final ExecutorService exec = Executors.newFixedThreadPool(4, daemonThreadFactory); 
    
    private boolean hasNext = false;
    private Future<TensorPair> buf;
    
    protected final TensorIter iter;
    protected final Engine eg;
    protected final int batch_size;
    
    public BufferedTensorIter(TensorIter iter, Engine eg, int batch_size) {
        this.eg = eg;
        this.iter = iter;
        this.batch_size = batch_size;
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-functions">
    public Engine engine() { return eg; }
    public int batch_size() { return batch_size; }
    public TensorIter iter() { return iter; }
    
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName());
        sb.append("{ engine = ").append(eg.getClass());
        sb.append(", batch_size = ").append(batch_size).append(" }");
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(128);
        this.append(sb);
        return sb.toString();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    public BufferedTensorIter shuffle_sort() { iter.shuffle_sort(); return this; }  
    public BufferedTensorIter shuffle_swap(double percent) { iter.shuffle_swap(percent); return this; } 
    public BufferedTensorIter shuffle_swap() { iter.shuffle_swap(); return this; }
    
    public BufferedTensorIter reset() { iter.reset(); hasNext = false; buf = null; return this; }
    public boolean hasNext() { return iter.hasNext() || hasNext; }
    
    public TensorPair next() { return next(false); }
    public TensorPair next(boolean random){//batch_size
        if(buf == null) {//buf == null, the first time to load TensorPair
            if(!iter.hasNext()) return null;
            buf = exec.submit(()->{ return iter.next(eg, batch_size, random); });
        }
        
        TensorPair kv; try { kv = buf.get(); } 
        catch(InterruptedException | ExecutionException e) { 
            throw new RuntimeException(e); 
        }
       
        hasNext = iter.hasNext();//if iter.hasNext = true: preload the next batch 
        if(hasNext) buf = exec.submit(()-> { return iter.next(eg, batch_size, random); });
        return kv;
    }
    //</editor-fold>
}
