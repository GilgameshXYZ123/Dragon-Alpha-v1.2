/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.db.jdbc.connection;

import java.sql.Connection;
import z.util.db.NewConnection;

/**
 *
 * @author dell
 */
public interface JDBCNC extends NewConnection<Connection>
{
    //<editor-fold defaultstate="collapsed" desc="Properties">
    public static final String DOCTYPE_JDBC_SOURCE="JDBC-SOURCE";
    
    public static final String DRIVER="jdbc.driver";
    public static final String URL="jdbc.url";
    
    public static final String QUERY="jdbc.query";
    public static final String PASSWORD="jdbc.password";
    public static final String USER="jdbc.user";
    
    public static final String CONNECT_RETRY_INTERVAL="jdbc.connect.retry.interval";
    public static final String CONNECT_RETRY_MAX_INTERVAL="jdbc.connect.retry.max.interval";
    public static final String CONNECT_RETRY_TIMES="jdbc.connect.retry.times";
    public static final String CONNECT_RETRY_BACKOFF="jdbc.connect.retry.backoff";
    //</editor-fold>
    
    public String getDriver();
    public String getUrl();
    public String getUser();
    public String getPassword();
}
