/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.pool;

import java.util.concurrent.CountDownLatch;
import z.util.factory.Meta;
import z.util.lang.Lang;
import z.util.lang.annotation.Passed;
import z.util.lang.exception.IAE;
import z.util.pool.exception.ClearException;
import z.util.pool.exception.CreateException;
import z.util.pool.exception.DestroyException;
import z.util.pool.imp.PoolBuffer;
import z.util.pool.imp.PoolManager;

/**
 * <pre>
 * This is an Extensive Implementation of Pool.
 * (1)to talk about the init params.
 * =>waitTime: The longest waiting time for a thread to get a unit of resource.
 * =>Manager: {@link PoolManager}
 * =>Buffer: {@link PoolBuffer}
 * 
 * (2)workFlow:
 * =>init: the initSize=(int)Math.ceil(fixedUpLimit*initProp);
 * Obviously, the pool only add a proportion of fixedUpLimit, as if it adds
 * all the resources, it way cost so long time.
 * the remained part will be added during the worktime.
 * if it fails to some resources, an Exception will be throwed.
 * 
 * =>getResource:
 * {@code
 *  if pool.isEmpty(): waitForResource()
 *  else return a unit a of resource.
 * }
 * 
 * =>returnResource
 * {@code
 * if pool.avaiableNumber>=size: 
 *  destroy the resource, otherwise the pool will be overflowing.
 * else: check if the resource is useable
 *     if not create a new resource.
 *     add the reosource to the pool.
 * }
 * 
 * =>clear:
 * if it fails to destroy resources, an Exception will be throwed.
 * you can clear a Pool first, and call init to reuse the Pool.
 * 
 * </pre>
 * @author dell
 * @param <M> the type of the PoolManager
 * @param <T> the type of resources
 * @param <B> the collection to buffer the resources
 */
@Passed
public class FixedPool<M extends PoolManager<T>, T, B extends PoolBuffer<T>> implements Pool<T>
{
    private boolean inited;
    
    protected M manager;
    protected B buffer;
    
    protected long waitTime;
    protected final double initProp;
    protected final int fixedLimit;
    
    protected int size;
    
    protected FixedPool(M manager, B buffer, long waitTime, int fixedLimit, double initProp)
    {
        if(manager==null) throw new NullPointerException();
        if(waitTime<=0) throw new IAE("wait time must postive");
        if(fixedLimit<=0) throw new IAE("initSize must postive");
        if(initProp>1||initProp<0) throw new IAE("initProp must between 0 and 1");
        
        this.manager=manager;
        this.buffer=buffer;
        this.waitTime=waitTime;
        this.fixedLimit=fixedLimit;
        this.initProp=initProp;
        this.size=0;
        this.inited=false;
    }
    protected FixedPool(Meta meta)
    {
        this(meta.getValue(Pool.MANAGER),
             meta.getValue(Pool.BUFFER),
             meta.getValue(Pool.WAIT_TIME_MS),
             meta.getValue(Pool.FIXED_LIMIT),
             meta.getValue(Pool.INIT_PROPORTION));
    }
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    public long getWaitTime() 
    {
        return waitTime;
    }
    @Override
    public M getManager()
    {
        return manager;
    }
    @Override
    public B getBuffer()
    {
        return buffer;
    }
    public int getFixedLimit()
    {
        return fixedLimit;
    }
    public double initProportion()
    {
        return initProp;
    }
    @Override
    public synchronized int size()
    {
        return size;
    }
    public synchronized int number() 
    {
        return buffer.number();
    }
    public synchronized int borrowedNumber()
    {
        return fixedLimit-buffer.number();
    }
    public synchronized boolean isEmpty() 
    {
        return buffer.isEmpty();
    }
    public void append(StringBuilder sb)
    {
        sb.append('[').append(this.getClass()).append("]");
        sb.append("\n\tmanger = ").append(manager.getClass());
        sb.append("\n\tbuffer = ").append(buffer.getClass());
        sb.append("\n\twaitTime = ").append(waitTime);
        sb.append("\n\tinitSize = ").append(fixedLimit);
        sb.append("\n\tinitProp = ").append(initProp);
    }
    @Override
    public String toString()
    {
        StringBuilder sb=new StringBuilder();
        this.append(sb);
        return sb.toString();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Inner-Operator">
    protected synchronized T waitForReource() throws InterruptedException
    {
        if(size<fixedLimit)
        {
            size++;
            return manager.create();
        }
        this.wait(waitTime);
        return buffer.remove();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Outer-Operator">
    @Passed
    @Override
    public synchronized void init()
    {
        if(this.inited) 
            throw new RuntimeException("The Pool has been inited Before, Clear it First");
        int count=0;
        buffer.clear();
        size=(int)Math.ceil(fixedLimit*initProp);
        
        for(int i=0;i<size;i++) 
        try
        {
            buffer.add(manager.create());
        }
        catch(CreateException e)
        {
            count++;
            e.printStackTrace();
        }
        
        if(buffer.size()==0) 
            throw new CreateException("Fail to init the Pool, as there has no resource");
        if(count!=0)
            throw new CreateException("Failt to create "+count+" unit of resource");
        inited=true;
    }
    @Passed
    @Override
    public synchronized void clear()
    {
        if(!inited)
            throw new RuntimeException("Unable to clear an uninitialized Pool");
        int count=0;
        for(T res:buffer)
        try 
        {
            manager.destroy(res);
        }
        catch(DestroyException e)
        {
            count++;
            e.printStackTrace();
        }
        buffer.clear();
        if(count!=0) 
            throw new ClearException("Fail to destroy "+count+" unit of resource");
        inited=false;
    }
    @Passed
    @Override
    public synchronized T getResource() throws InterruptedException 
    {
        return (buffer.isEmpty()? this.waitForReource(): buffer.remove());
    }
    @Passed
    @Override
    public synchronized boolean returnResource(T res) throws Exception
    {
        if(buffer.number()>=size)
        {
            manager.destroy(res);
            return false;
        }
        buffer.add(manager.check(res)? res: manager.create());
        this.notify();
        return true;
    }
    //</editor-fold>
}
