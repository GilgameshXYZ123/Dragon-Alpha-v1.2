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
import org.w3c.dom.Node;
import z.util.factory.Meta;
import z.util.pool.exception.CreateException;
import z.util.xml.XML;
/**
 *
 * @author dell
 */
public class SingleConnection implements JDBCNC
{
    //columns-------------------------------------------------------------------
    protected String driver;
    protected String url;
    protected String user;
    protected String password;
    
    protected int retryTimes=1;
    protected long retryInterval=100;//ms
    
    protected Connection con;
    protected volatile boolean mutex=false;
    
    //constructors--------------------------------------------------------------
    public SingleConnection(Meta columns) throws Exception
    {
        //basic-----------------------------------------------------------------
        driver=columns.getValueNoNull(JDBCNC.DRIVER);
        url=columns.getValueNoNull(JDBCNC.URL);
        user=columns.getValueNoNull(JDBCNC.USER);
        password=columns.getValueNoNull(JDBCNC.PASSWORD);
        String query=columns.getValue(JDBCNC.QUERY);
        if(query!=null) url=url+'?'+query;
        Class.forName(driver);
        
        //extensive-------------------------------------------------------------
        retryTimes=columns.getValueOrDefault(JDBCNC.CONNECT_RETRY_TIMES, retryTimes);
        retryInterval=columns.getValueOrDefault(JDBCNC.CONNECT_RETRY_INTERVAL, retryInterval);
    }
    //basic---------------------------------------------------------------------
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
    public int getRetryTimes() 
    {
        return retryTimes;
    }
    public long getRetryInterval() 
    {
        return retryInterval;
    }
    @Override
    public String toString()
    {
        StringBuilder sb=new StringBuilder();
        sb.append("SingleConnection = {");
        sb.append("\n\tdriver = ").append(driver);
        sb.append("\n\t url = ").append(url);
        sb.append("\n\t user = ").append(user);
        sb.append("\n\t password = ").append(password);
        sb.append("\n\t retryTimes = ").append(retryTimes);
        sb.append("\n\t retryInterval = ").append(retryInterval);
        sb.append("\n}");
        return sb.toString();
    }
    //functions-----------------------------------------------------------------
    private void initConnection()
    {
        try
        {
            con=DriverManager.getConnection(url, user, password);
            if(con!=null) return;
        }
        catch(SQLException e)
        {
            try
            {
                if(retryInterval!=0) Thread.sleep(retryInterval);
            }
            catch(InterruptedException ex){throw new CreateException(ex);}
        }
        throw new CreateException("Fail to getConnection");
    }
    @Override
    public synchronized Connection newConnection() throws Exception 
    {
        if(this.con==null) 
        {
            this.initConnection();
            return con;
        }
        if(!mutex) throw new RuntimeException("The Mutex Conection has been occupied");
        mutex=false;
        return con;
    }
    @Override
    public synchronized void releaseConnection(Connection con) throws Exception 
    {
        mutex=true;
    }
    public synchronized void close() throws SQLException 
    {
        if(con!=null) 
        {
            con.close();
            con=null;
            mutex=false;
        }
    }
    //builder-------------------------------------------------------------------
    public static SingleConnection valueOf(String src) throws Exception
    {
        return valueOf(XML.getDocument(src));
    }
    public static SingleConnection valueOf(InputStream in) throws Exception
    {
        return valueOf(XML.getDocument(in));
    }
    public static SingleConnection valueOf(Document dou) throws Exception
    {
        XML.checkDocType(dou, JDBCNC.DOCTYPE_JDBC_SOURCE);
        Node conf=dou.getElementsByTagName("configuration").item(0);
        Meta columns=Meta.valueOf(conf.getChildNodes());
        SingleConnection cf=new SingleConnection(columns);
        return cf;
    }
}
