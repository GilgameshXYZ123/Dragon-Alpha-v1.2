/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.db.jdbc.connection;

import java.io.InputStream;
import java.sql.Connection;
import org.w3c.dom.Document;
import z.util.factory.Meta;
import z.util.lang.annotation.Passed;
import z.util.pool.ElasticPool;
import z.util.pool.LinkedBuffer;
import z.util.pool.Pool;
import z.util.xml.XML;

/**
 *
 * @author dell
 */
@Passed
public class ConnectionPool implements JDBCNC
{
    protected ConnectionFactory manager;
    protected Pool<Connection> pool;
    
    protected ConnectionPool() {}
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    @Override
    public String getDriver() 
    {
        return manager.driver;
    }
    @Override
    public String getUrl() 
    {
        return manager.url;
    }
    @Override
    public String getUser() 
    {
        return manager.user;
    }
    @Override
    public String getPassword() 
    {
        return manager.password;
    }
    public ConnectionFactory getConnectionFactory()
    {
        return manager;
    }
    public Pool getPool()
    {
        return pool;
    }
    public void append(StringBuilder sb)
    {
        sb.append("ConnectionPool = {\n ");
        sb.append("[ConnectionFacty]\n").append(manager.toString());
        sb.append('\n').append(pool.toString());
        sb.append("\n}");
    }
    @Override
    public String toString()
    {
        StringBuilder sb=new StringBuilder();
        this.append(sb);
        return sb.toString();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Outer-Operator">
     @Override
    public Connection newConnection() throws Exception 
    {
        return pool.getResource();
    }
    @Override
    public void releaseConnection(Connection con) throws Exception 
    {
        pool.returnResource(con);
    }
    public void init()
    {
        pool.init();
    }
    public void clear()
    {
        pool.clear();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Buillder">
    public static ConnectionPool valueOf(String src) throws Exception
    {
        return valueOf(XML.getDocument(src));
    }
    public static ConnectionPool valueOf(InputStream in) throws Exception
    {
        return valueOf(XML.getDocument(in));
    }
    public static ConnectionPool valueOf(Document dou) throws Exception
    {
        Meta meta=Meta.valueOf(dou, DOCTYPE_JDBC_SOURCE, "configuration");
        ConnectionPool pool=new ConnectionPool();
        pool.manager=new ConnectionFactory(meta);
        
        meta.put(Pool.MANAGER, pool.manager);
        meta.putIfAbsent(Pool.BUFFER, new LinkedBuffer());
        meta.putIfAbsent(Pool.CLASS, ElasticPool.class);
        
        pool.pool=Pool.valueOf(meta);
        return pool;
    }
    public static ConnectionPool valueOf(ConnectionFactory confac, String src) throws Exception
    {
        return valueOf(confac, XML.getDocument(src));
    }
    public static ConnectionPool valueOf(ConnectionFactory confac, InputStream in) throws Exception
    {
        return valueOf(confac, XML.getDocument(in));
    }
    public static ConnectionPool valueOf(ConnectionFactory confac, Document dou) throws Exception
    {
        if(confac==null) throw new NullPointerException("ConnectionFactory");
        
        Meta meta=Meta.valueOf(dou, DOCTYPE_JDBC_SOURCE, "configuration");
        ConnectionPool pool=new ConnectionPool();
        pool.manager=confac;
        
        meta.put(Pool.MANAGER, pool.manager);
        meta.putIfAbsent(Pool.BUFFER, new LinkedBuffer());
        meta.putIfAbsent(Pool.CLASS, ElasticPool.class);
        
        pool.pool=Pool.valueOf(meta);
        return pool;
    }
    //</editor-fold>
}
