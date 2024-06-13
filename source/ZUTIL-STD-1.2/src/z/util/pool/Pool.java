/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.pool;

import java.lang.reflect.Constructor;
import java.util.concurrent.CountDownLatch;
import z.util.factory.Meta;
import z.util.lang.Lang;
import z.util.lang.annotation.Passed;
import z.util.lang.exception.IAE;
import z.util.pool.imp.PoolBuffer;
import z.util.pool.imp.PoolManager;

/**
 *
 * @author dell
 * @param <T>
 */
public interface Pool<T>
{
    //<editor-fold defaultstate="collapsed" desc="Properties">
    public static final String CLASS=          "pool.class";  
    public static final String MANAGER=        "pool.manager";
    public static final String BUFFER=         "pool.buffer";
    public static final String WAIT_TIME_MS=   "pool.wait.time";
    public static final String INIT_PROPORTION="pool.init.propotion";
    public static final String FIXED_LIMIT=    "pool.fixed.limit";
    
    public static final String ELASTIC_LIMIT=              "pool.elastic.limit";
    public static final String ELASTIC_INCREASE_NUM=       "pool.elastic.increase.num";
    public static final String ELASTIC_EXPIRE_FREQUENCY_MS="pool.elastic.expire.frequency";
    //</editor-fold>
    
    public void init();
    public void clear();
    public int size();
    public T getResource() throws Exception;
    public boolean returnResource(T res) throws Exception;
    public PoolManager getManager();
    public PoolBuffer getBuffer();
    
    public static <T extends Pool> T valueOf(Meta meta) throws Exception
    {
        //check the pool clazz--------------------------------------------------
        Object val=meta.get(CLASS);
        Class clazz=null;
        if(val==null) clazz=ElasticPool.class;
        else if(val instanceof String) clazz=Class.forName((String) val);
        else if(val instanceof Class) clazz=(Class) val;
        else throw new IAE("Wrong type of Property:"+CLASS);
        
        //check the pool manager------------------------------------------------
        val=meta.get(MANAGER);
        if(val instanceof PoolManager);
        else if(val==null) throw new NullPointerException("PropertyMissed:"+MANAGER);
        else if(val instanceof String) meta.put(MANAGER, Class.forName((String)val).newInstance());
        else if(val instanceof Class) meta.put(MANAGER, ((Class)val).newInstance());
        else throw new IAE("Wrong type of Property:"+MANAGER);
        
        //check the pool buffer-------------------------------------------------
        val=meta.get(BUFFER);
        if(val instanceof PoolBuffer);
        else if(val==null) meta.put(BUFFER, new LinkedBuffer());
        else if(val instanceof String)  meta.put(BUFFER, Class.forName((String) val).newInstance());
        else if(val instanceof Class) meta.put(BUFFER, ((Class)val).newInstance());
        else throw new IAE("Wrong type of Property:"+MANAGER); 
        
        //create the pool-------------------------------------------------------
        Constructor con=clazz.getConstructor(Meta.class);
        T pool=(T) con.newInstance(meta);
        return pool;
    }
    //<editor-fold defaultstate="collapsed" desc="Pool-Test-Code">
    public static final PoolManager TM=new PoolManager() 
    {
        @Override
        public Object create() {return Lang.exr().nextInt();}
        @Override
        public boolean check(Object val) {return val!=null;}
        @Override
        public void destroy(Object val) {}
    };
    @Passed
    public static void testPool(int concurrent, long sleepTime, int round, Pool pool)
    {
        if(pool==null) throw new NullPointerException();
        try
        {
            pool.init();
            CountDownLatch cdl=new CountDownLatch(concurrent);
            Thread[] t=new Thread[concurrent];
            for(int i=0;i<t.length;i++)
            t[i]=new Thread(new Runnable() 
            {
                private long id=-1;
                @Override
                public void run() 
                {
                    for(int j=0,len=(round>>2)+Lang.exr().nextInt(round>>2);j<len;j++)
                    try
                    {
                        id=Thread.currentThread().getId();
                        Object x=pool.getResource();
                        System.err.println("pool-size="+pool.size()+" "+id+" get :"+x);
                        Thread.sleep(sleepTime);
                        pool.returnResource(x);
                        System.err.println("pool-size="+pool.size()+" "+id+" return :"+x);
                    }
                    catch(Exception e)
                    {
                        e.printStackTrace();
                    }
                    cdl.countDown();
                }
            });
            for (Thread t1 :t) t1.start();
            cdl.await();
            System.err.println("END");
            pool.clear();
        }
        catch(InterruptedException e)
        {
            e.printStackTrace();
        }
    }
    //</editor-fold>
}
