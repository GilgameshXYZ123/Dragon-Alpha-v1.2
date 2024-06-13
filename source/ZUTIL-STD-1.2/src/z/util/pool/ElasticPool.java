/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.pool;

import z.util.concurrent.BinarySemaphore;
import z.util.factory.Meta;
import z.util.lang.annotation.Passed;
import z.util.lang.exception.IAE;
import z.util.pool.exception.DestroyException;
import z.util.pool.imp.PoolBuffer;
import z.util.pool.imp.PoolManager;

/**
 *
 * @author dell
 * @param <M>
 * @param <T>
 * @param <B>
 */
@Passed("2021/4/19")
public class ElasticPool<M extends PoolManager<T>, T, B extends PoolBuffer<T>>  extends FixedPool<M , T, B>
{
    //<editor-fold defaultstate="collapsed" desc="class Expirer">
    protected class Expirer implements Runnable
    {
        @Override
        public void run()
        {
            while(running)
            {
                //don't worry, if the resource is borrowed, its not in the pool
                if(size<=fixedLimit) full.P();
                else 
                {
                    try
                    {
                        if(!buffer.isEmpty())
                        synchronized(ElasticPool.this)
                        {
                            manager.destroy(buffer.remove());
                            size--;
                        }
                        full.sleep(waitTime);
                    }
                    catch(DestroyException e){}
                }
            }
        }
    }
    //</editor-fold>
    protected final int elasticLimit;//minSize=initSize
    protected int increaseNum;
    protected long expireFrequency;
    
    protected BinarySemaphore full;
    protected Expirer expirer;
    protected Thread ext;
    protected boolean running=false;

    public ElasticPool(M manager, B buffer, long waitTime, int fixedUpLimit, double initProp,
            int elasticLimit, int increaseNum, long expireFrequency) 
    {
        super(manager, buffer, waitTime, fixedUpLimit, initProp);
        
        if(expireFrequency<=0) throw new IAE("expirerTime must be postive");
        
        if(increaseNum<=0) throw new IAE("increaseNum must be positive");
        
        if(elasticLimit<fixedUpLimit) 
        {
            elasticLimit=(int) (fixedUpLimit*1.2);
            if(elasticLimit==fixedUpLimit) elasticLimit=fixedUpLimit+increaseNum*3;
        }
        
        this.full=new BinarySemaphore();
        this.elasticLimit=elasticLimit;
        this.size=0;
        this.increaseNum=increaseNum;
        this.expireFrequency=expireFrequency;
    }
    public ElasticPool(Meta meta)
    {
        this(meta.getValue(Pool.MANAGER),
             meta.getValue(Pool.BUFFER),
             meta.getValue(Pool.WAIT_TIME_MS),
             meta.getValue(Pool.FIXED_LIMIT),
             meta.getValue(Pool.INIT_PROPORTION),
             
             meta.getValue(Pool.ELASTIC_LIMIT),
             meta.getValue(Pool.ELASTIC_INCREASE_NUM),
             meta.getValue(Pool.ELASTIC_EXPIRE_FREQUENCY_MS));
    }
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    public int getElasticLimit() 
    {
        return elasticLimit;
    }
    public int getIncreaseNum() 
    {
        return increaseNum;
    }
    public long getExpireFrequency() 
    {
        return expireFrequency;
    }
    @Override
    public void append(StringBuilder sb)
    {
        super.append(sb);
        sb.append("\n\telasticLimit = ").append(elasticLimit);
        sb.append("\n\tincreaseNum = ").append(increaseNum);
        sb.append("\n\texpireFrequency = ").append(expireFrequency);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Inner-Operator">
    @Override
    protected T waitForReource() throws InterruptedException
    {
        if(size<fixedLimit)
        {
            size++;
            return manager.create();
        }
        if(size<elasticLimit)
        {
            int div=elasticLimit-size, len=(increaseNum<div? increaseNum:div);
            for(int i=1;i<len;i++) buffer.add(manager.create());
            size+=len;
            full.V();
            return manager.create();
        }
        this.wait(waitTime);
        return buffer.remove();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Outer-Operator">
    @Override
    public void init()
    {
        super.init();
        synchronized(this)
        {
            ext=new Thread(expirer=new Expirer());
            ext.setDaemon(true);//full.V();//may be the expirer is sleep
            
            running=true;
            ext.start();
        }
    }
    @Override
    public void clear()
    {
        super.clear(); 
        synchronized(this)
        {
            running=false;
            full.V();
            
            size=0;
            expirer=null;
            ext=null;
        }
    }
     //</editor-fold>
}
