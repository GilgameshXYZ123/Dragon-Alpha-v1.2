/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.pool.imp;

import z.util.pool.FixedPool;

/**
 *
 * @author dell
 * @param <R>
 * @param <T>
 */
public class PoolRunnable<R,T> implements Runnable
{
    //columns-------------------------------------------------------------------
    private R result;
    private T resource;
    private FixedPool<?,T,?> pool;
    private Processor<R,T> processor;
    
    //constructor---------------------------------------------------------------
    public PoolRunnable(R result, T resource, FixedPool pool, Processor<R, T> processor)    
    {
        this.result = result;
        this.resource = resource;
        this.pool = pool;
        this.processor = processor;
    }
    //operator------------------------------------------------------------------
    @Override
    public void run() {
        try
        {
            processor.process(result, resource);
            pool.returnResource(resource);
        }
        catch(Exception e)
        {
            throw new RuntimeException(e);
        }
    }
}
