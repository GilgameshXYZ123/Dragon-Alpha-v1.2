/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.db.jdbc.connection;

import java.io.InputStream;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import org.w3c.dom.Document;
import z.util.factory.Meta;
import z.util.lang.annotation.Passed;
import z.util.pool.exception.CreateException;
import z.util.xml.XML;
import z.util.pool.imp.PoolManager;

/**
 *
 * @author dell
 */
@Passed
public class ConnectionFactory implements PoolManager<Connection>, JDBCNC
{
    protected String driver;
    protected String url;
    protected String user;
    protected String password;
    
    protected int retryTimes;
    protected long retryInterval;//ms
    protected long retryBackoff;//ms
    protected long retryMaxInterval;
    
    protected ConnectionFactory(Meta meta) throws Exception
    {
        //basic-----------------------------------------------------------------
        driver=meta.getValueNoNull(DRIVER);
        url=meta.getValueNoNull(URL);
        user=meta.getValueNoNull(USER);
        password=meta.getValueNoNull(PASSWORD);
        String query=meta.getValue(QUERY);
        if(query!=null) url=url+'?'+query;
        else throw new NullPointerException("query");
        Class.forName(driver);
        
        //extensive-------------------------------------------------------------
        retryTimes=meta.getValueOrDefault(CONNECT_RETRY_TIMES, 1);
        retryInterval=meta.getValueOrDefault(CONNECT_RETRY_INTERVAL, 100);
        retryMaxInterval=meta.getValueOrDefault(CONNECT_RETRY_MAX_INTERVAL, -1);
        retryBackoff=meta.getValueOrDefault(CONNECT_RETRY_BACKOFF, retryInterval*0.2);
    }
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    @Override
    public String getDriver() 
    {
        return driver;
    }
    @Override
    public String getUrl() 
    {
        return url;
    }
    @Override
    public String getUser() 
    {
        return user;
    }
    @Override
    public String getPassword() 
    {
        return password;
    }
    public void append(StringBuilder sb)
    {
        sb.append("ConnectionFactory = {");
        sb.append("\n\tdriver = ").append(driver);
        sb.append("\n\turl = ").append(url);
        sb.append("\n\tuser = ").append(user);
        sb.append("\n\tpassword = ").append(password);
        sb.append("\n\tretryTimes = ").append(retryTimes);
        sb.append("\n\tretryInterval = ").append(retryInterval);
        sb.append("\n\tretryMaxInterval = ").append(retryMaxInterval);
        sb.append("\n\tretryBackoff = ").append(retryBackoff);
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
    //<editor-fold defaultstate="collapsed" desc="NewConnection">
    @Override
    public Connection newConnection()
    {
        return this.create();
    }
    @Override
    public void releaseConnection(Connection con) throws SQLException 
    {
        if(con!=null) con.close();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="PoolManager">
    @Override
    public Connection create() 
    {
        Connection con=null;
        long sleep=retryInterval;
        for(int i=0;i<this.retryTimes;i++)
        try
        {
            con=DriverManager.getConnection(url, user, password);
            if(con!=null) return con;
        }
        catch(SQLException e) 
        {
            System.err.print(e);
            if(retryInterval!=0) 
            try 
            {
                Thread.sleep(sleep);
            }
            catch(InterruptedException ex)
            {throw new CreateException(ex);}
            sleep+=retryBackoff;
            if(retryMaxInterval!=-1&&sleep>retryMaxInterval) 
                sleep=retryMaxInterval;
        }
        throw new CreateException("Fail to getConnection");
    }
    @Override
    public boolean check(Connection val)
    {
        if(val==null) return false;
        try
        {
            return !val.isClosed();
        }
        catch(SQLException e)
        {
            return false;
        }
    }
    @Override
    public void destroy(Connection val)
    {
        if(val==null) return;
        try
        {
            if(!val.isClosed()) val.close();
        }
        catch(SQLException e)
        {
            e.printStackTrace();
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Builder">
    public static ConnectionFactory valueOf(Meta meta) throws Exception
    {
        return new ConnectionFactory(meta);
    }
    public static ConnectionFactory valueOf(Document dou) throws Exception
    {
        return new ConnectionFactory(Meta.valueOf(dou, DOCTYPE_JDBC_SOURCE, "configuration"));
    }
    public static ConnectionFactory valueOf(String src) throws Exception
    {
        return valueOf(XML.getDocument(src));
    }
    public static ConnectionFactory valueOf(InputStream in) throws Exception
    {
        return valueOf(XML.getDocument(in));
    }
    //</editor-fold>
}
