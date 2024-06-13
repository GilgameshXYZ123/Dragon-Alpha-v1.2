/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.lang;

import z.util.factory.Meta;
import z.util.math.vector.Vector;

/**
 * 
 * @author Gilgamesh
 */
public class SimpleTimer 
{
    public static SimpleTimer instance() { return new SimpleTimer(); }
    public static SimpleTimer clock() {  return new SimpleTimer().record(); }
    
    protected long last = -1;
    protected long current;
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public long last_time_millis() { return last; }
    public long current_time_millis() { return current; }
    
    public void appand(StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append("{ ");
        sb.append("last_time_millis = ").append(last);
        sb.append("current_time_millis = ").append(current);
        sb.append("milli seconds = ").append(timeStamp_dif_millis());
        sb.append(" }");
    }
    
    public long natoTime() { return System.nanoTime(); }
    public long currentTimeMillis() { return System.currentTimeMillis(); }
    
    @Override
    public String toString() { 
        StringBuilder sb = new StringBuilder(128);
        this.appand(sb);
        return sb.toString();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="operations">
    public SimpleTimer record() {
        last = current;
        current = System.currentTimeMillis();
        return this; 
    }

    public long timeStamp_dif_millis() {  
        if(last <= 0) throw new RuntimeException("Please record at least twice, to find the difference");
        return current - last; 
    }
    
    private static final String TEST_AVG_TIME = "test.avg.time";
    private static final String TEST_STDDEV_TIME = "test.stddev.time";
    private static final String TEST_EACH_TIME = "test.each.time";
    
    public Meta test(int times, Runnable task) {
        long[] t = new long[times];
        for(int i=0; i<times; i++) {
            this.record();
            task.run();
            this.record();
            t[i] = this.timeStamp_dif_millis();
        }
        Meta mt = new Meta();
        mt.put(TEST_EACH_TIME, t);
        mt.put(TEST_AVG_TIME, Vector.average(t));
        mt.put(TEST_STDDEV_TIME, Vector.stddev(t));
        return mt;
    }
    //</editor-fold>
}
