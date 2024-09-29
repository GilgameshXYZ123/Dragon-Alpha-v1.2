/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;

/**
 *
 * @author Gilgamesh
 */
public interface Syncer {
    public void sync();
    
    //<editor-fold defaultstate="collapsed" desc="class: NullSyncer">
    public static final Syncer none = new NullSyncer();
    
    public static final class NullSyncer implements Syncer  {
        private NullSyncer() {}
        @Override public final void sync() {}
    }
    //</editor-fold>
    public static Syncer none() { return none; }
    
    //<editor-fold defaultstate="collapsed" desc="class: DualSyncer">
    public static final class DualSyncer implements Syncer 
    {
        private final Syncer sc1;
        private final Syncer sc2;
        
        public final Syncer sc1() { return sc1; }
        public final Syncer sc2() { return sc2; }
        
        public DualSyncer(Syncer sc1, Syncer sc2) {
            this.sc1 = sc1;
            this.sc2 = sc2;
        }
        
        @Override
        public final void sync() {
            if (sc1 != null) sc1.sync();
            if (sc2 != null) sc2.sync();
        }
    }
    //</editor-fold>
    public static Syncer dual(Syncer sc1, Syncer sc2) { return new DualSyncer(sc1, sc2); }
    
    //<editor-fold defaultstate="collapsed" desc="class: ChainSyncer">
    public static final class ChainSyncer implements Syncer 
    {
        private final Syncer[] scs;
        
        public final Syncer[] scs() { return scs; }
        public final Syncer sc(int index) { 
            if(index < 0) index = scs.length - 1;
            return scs[index];
        }
        
        public ChainSyncer(Syncer... syncers) {  this.scs = syncers;  }
        
        @Override 
        public final void sync() {
            if(scs == null) return;
            for(Syncer sc : scs) if(sc != null) sc.sync();
        }
    }
    //</editor-fold>
    public static Syncer chain(Syncer... syncers) { return new ChainSyncer(syncers); }
  
    //<editor-fold defaultstate="collapsed" desc="class: OneCalledSyncer">
    public static final class OnceCalledSyncer implements Syncer 
    {
        private final Syncer sc;
        private boolean called = false;
        
        public boolean called() { return called; }
        public Syncer sc() { return sc; }
        
        public OnceCalledSyncer(Syncer sc) {  this.sc = sc;  }
        
        @Override 
        public void sync() {
            synchronized(this) {
                if(!called && sc != null ) { sc.sync(); called = true; }
            }
        }
    }
    //</editor-fold>
    public static Syncer onceCalled(Syncer sc) { return new OnceCalledSyncer(sc); }
    
    //<editor-fold defaultstate="collapsed" desc="static class: RemoteSync">
    public static class RemoteSync {
        private static final ThreadFactory daemonThreadFactory = new ThreadFactory() {
            @Override
            public Thread newThread(Runnable r) {
                Thread t = new Thread(r);
                t.setDaemon(true); 
                return t;
            }
        };
        private static final ExecutorService exec = Executors.newFixedThreadPool(2, daemonThreadFactory);
        
        /**
         * <pre>
         * Remote Sync to wait computation result and Release Resource in another thread.
         * As Tensor.c() is an synchronized method, for a specific Tensor ts, when 
         * one thread is calling ts.c(), another thread needs to wait this call finished
         * to call ts.c();
         * For VGG16: 
         * remote = true : loss: 2.3 -> 1.6, time = 1.502, 1406 MB
         * remore = false: loss: 2.3 -> 1.6, time = 1.499, 1406 MB
         * </pre>
         * @param ts 
         */
        public static void sync(Tensor ts) {
            if (ts.eg.sync()) { ts.c(); return; }
            synchronized(ts) {
                if (ts.syncer == null) return;
                Syncer sc = ts.syncer;
                ts.syncer = new FutureSyncer<>(exec.submit(() -> { sc.sync(); }));
            }
        }
        
        public static void delete(Tensor ts) {
            if (ts.eg.sync) { ts.c(); ts.delete(); return; }
            synchronized(ts) {
                if (ts.syncer == null) { ts.delete(); return; }    
                Syncer sc = ts.syncer;
                ts.syncer = new FutureSyncer<>(exec.submit(()-> { sc.sync(); ts.delete(); }));
            }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: FutureSyncer">
    public static class FutureSyncer<T> implements Syncer {
        private final Future<T> ft;
        
        public FutureSyncer(Future<T> future) { this.ft = future; }
        
        @Override
        public final void sync() {
            try { ft.get(); }
            catch(InterruptedException | ExecutionException e) {
                throw new RuntimeException(e);
            }
        }
    }
    //</editor-fold>
    public static<T> Syncer future(Future<T> future) { return new FutureSyncer<>(future); }
    
    //<editor-fold defaultstate="collapsed" desc="class: FollowSyncer">
    public static class FollowSyncer implements Syncer
    {
        private final Tensor ts;
        
        public FollowSyncer(Tensor ts) { this.ts = ts; }
        
        @Override
        public final void sync() { 
            if(ts != null) ts.c(); 
        }
    }
    //</editor-fold>
    public static Syncer follow(Tensor ts) { return new FollowSyncer(ts); }
}
