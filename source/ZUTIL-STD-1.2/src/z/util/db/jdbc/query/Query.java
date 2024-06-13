/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.db.jdbc.query;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Objects;
import z.util.db.NewConnection;
import z.util.ds.linear.ZArrayList;
import z.util.function.Creator;
import z.util.lang.annotation.Passed;
import z.util.math.vector.Vector;

/**
 *
 * @author dell
 */
public class Query 
{
    //<editor-fold defaultstate="collapsed" desc="static class QueryParserMap">
    final static class QueryParserMap extends HashMap<Class, QueryParser>
    {
        public QueryParser getParserOrLoad(Class key)
        {
            QueryParser qp=this.get(key);
            if(qp==null)
                try
                {
                    this.put(key, qp=new QueryParser(key));
                }
                catch(Exception e) {throw new RuntimeException(e);}
            return qp;
        }
    }
    //</editor-fold>
    NewConnection<Connection> nc;
    QueryParserMap map;
    
    public Query(NewConnection<Connection> nc)
    {
        Objects.requireNonNull(nc, "newConnection");
        this.nc=nc;
        map=new QueryParserMap();
    }
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    public NewConnection<Connection> getNC()
    {
        return nc;
    }
    public Connection newConnection() throws Exception
    {
        return nc.newConnection();
    }
    public void releaseConnection(Connection jedis) throws Exception
    {
        nc.releaseConnection(jedis);
    }
    public void append(StringBuilder sb)
    {
        sb.append("Query @ ").append(Integer.toHexString(this.hashCode()));
        sb.append("\n[NewConnection]\n").append(nc);
        sb.append("\n[QueryParserMap]\n");
        Vector.appendLn(sb, map);
    }
    @Override
    public String toString()
    {
        StringBuilder sb=new StringBuilder();
        this.append(sb);
        return sb.toString();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Create-QueryString">
    @Passed
    static String createSelectQuery(QueryParser qp)//select * from ?
    {
        return "select * from "+qp.tableName;
    }
    @Passed
    static String createSelectQuery(QueryParser qp, String condition)//select * from ? where ?
    {
        return "select * from "+qp.tableName+" "+condition;
    }
    static String createSelectQuery(QueryParser qp, String[] colName)
    {
        StringBuilder sb=new StringBuilder();
        sb.append("select ").append(colName[0]);
        for(int i=1;i<colName.length;i++) sb.append(',').append(colName[i]);
        sb.append(" from ").append(qp.tableName);
        return sb.toString();
    }
    static String createSelectQuery(QueryParser qp, String[] colName, String condition)
    {
        StringBuilder sb=new StringBuilder();
        sb.append("select ").append(colName[0]);
        for(int i=1;i<colName.length;i++) sb.append(',').append(colName[i]);
        sb.append(" from ").append(qp.tableName).append(" ").append(condition);
        return sb.toString();
    }
    @Passed
    static String createInsertQuery(QueryParser qp)
    {
        StringBuilder sb=new StringBuilder();
        sb.append("insert into ").append(qp.tableName).append(" (").append(qp.cfs[0].key);
        for(int i=1;i<qp.cfs.length;i++) sb.append(",").append(qp.cfs[i].key);
        sb.append(") values(?");
        for(int i=1;i<qp.cfs.length;i++) sb.append(",?");
        sb.append(')');
        return sb.toString();
    }
    @Passed
    static String createInsertQuery(QueryParser qp, String[] colName)
    {
        StringBuilder sb=new StringBuilder();
        sb.append("insert into ").append(qp.tableName).append(" (").append(colName[0]);
        for(int i=1;i<colName.length;i++) sb.append(",").append(colName[i]);
        sb.append(") values(?");
        for(int i=1;i<colName.length;i++) sb.append(",?");
        sb.append(')');
        return sb.toString();
    }
    @Passed
    static String createUpdateQuery(QueryParser qp)
    {
        StringBuilder sb=new StringBuilder();
        sb.append("update ").append(qp.tableName).append(" set ").append(qp.cfs[0].key).append(" = ?");
        for(int i=1;i<qp.cfs.length;i++)
            sb.append(" ,").append(qp.cfs[i].key).append(" = ? ");
        sb.append(" where ").append(qp.pk.key).append(" = ?");
        return sb.toString();
    }
    @Passed
    static String createUpdateQuery(QueryParser qp, String[] colName)
    {
        StringBuilder sb=new StringBuilder();
        sb.append("update ").append(qp.tableName).append(" set ").append(colName[0]).append(" = ?");
        for(int i=1;i<colName.length;i++)
            sb.append(" ,").append(colName[i]).append(" = ? ");
        sb.append(" where ").append(qp.pk.key).append(" = ?");
        return sb.toString();
    }
    @Passed
    static String createDeleteQuery(QueryParser qp) //delete from ? where ? = ?
    {
        return "delete from "+qp.tableName+" where "+qp.pk.key+" = ?";
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Outer-Operator">
    //<editor-fold defaultstate="collapsed" desc="Common-Function">
    public int selectOneInt(String query, String column)
    {
        Connection con=null;
        PreparedStatement pst=null;
        ResultSet rs=null;
        int result=-1;
        try
        {
            con=nc.newConnection();
            pst=con.prepareCall(query);
            rs=pst.executeQuery();
            if(rs.next()) result=rs.getInt(column);
        }
        catch(Exception e) {throw new RuntimeException(e);}
        finally
        {
            try
            {
                if(rs!=null) rs.close();
                if(pst!=null) pst.close();
                if(con!=null) nc.releaseConnection(con);
            }
            catch(Exception e) {throw new RuntimeException(e);}
        }
        return result;
    }
    public ZArrayList<Integer> selectInt(String query, String column)
    {
        Connection con=null;
        PreparedStatement pst=null;
        ResultSet rs=null;
        ZArrayList result=null;
        int index=0;
        try
        {
            con=nc.newConnection();
            pst=con.prepareCall(query);
            rs=pst.executeQuery();
            result=new ZArrayList();
            while(rs.next()) result.add(rs.getInt(column));
        }
        catch(Exception e) {throw new RuntimeException(e);}
        finally
        {
            try
            {
                if(rs!=null) rs.close();
                if(pst!=null) pst.close();
                if(con!=null) nc.releaseConnection(con);
            }
            catch(Exception e) {throw new RuntimeException(e);}
        }
        return result;
    }
    public <T> ZArrayList<T> select(String query, Object[] args, Creator<ResultSet> creator) 
    {
        Connection con=null;
        PreparedStatement pst=null;
        ResultSet rs=null;
        ZArrayList list=null;
        try
        {
            con=nc.newConnection();
            pst=con.prepareCall(query);
            for(int i=0;i<args.length;i++) pst.setObject(i+1, args[i]);
            rs=pst.executeQuery();
            list=new ZArrayList<>();
            while(rs.next()) list.add(creator.create(rs));
            
        }
        catch(Exception e)
        {
            throw new RuntimeException(e);
        }
        finally
        {
            try
            {
                if(rs!=null) rs.close();
                if(pst!=null) pst.close();
                if(con!=null) nc.releaseConnection(con);
            }
            catch(Exception e)
            {
                throw new RuntimeException(e);
            }
        }
        return list;
    }
    public void releaseConnection(ResultSet rs) throws Exception
    {
        if(rs==null) return;
        Statement st=rs.getStatement();
        rs.close();
        if(st==null) return;
        Connection con=st.getConnection();
        st.close();
        if(con==null) return;
        nc.releaseConnection(con);
    }
    public void releaseConnection(Connection con, ResultSet rs) throws Exception
    {
        if(rs!=null) rs.close();
        nc.releaseConnection(con);
    }
    public void releaseConnection(Connection con, PreparedStatement pst, ResultSet rs) throws Exception
    {
        if(rs!=null) rs.close();
        if(pst!=null) pst.close();
        nc.releaseConnection(con);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Ecapsulated:Select">
    @Passed
    public <T> T selectOne(Class clazz, String condition)
    {
        T val;
        Connection con=null;
        PreparedStatement pst=null;
        ResultSet rs=null;
        try
        {
            QueryParser qp=map.getParserOrLoad(clazz);
            con=nc.newConnection();
            pst=con.prepareStatement(Query.createSelectQuery(qp, condition));
            rs=pst.executeQuery();
            val=qp.parse(rs);
        }
        catch(Exception e) {throw new RuntimeException(e);}
        finally
        {
            try
            {
                if(rs!=null) rs.close();
                if(pst!=null) pst.close();
                if(con!=null) nc.releaseConnection(con);
            }
            catch(Exception e) {throw new RuntimeException(e);}
        }
        return val;
    }
    @Passed
    public <T> LinkedList<T> select(Class clazz)
    {
        LinkedList<T> list=null;
        Connection con=null;
        PreparedStatement pst=null;
        ResultSet rs=null;
        try
        {
            QueryParser qp=map.getParserOrLoad(clazz);
            con=nc.newConnection();
            pst=con.prepareStatement(Query.createSelectQuery(qp));
            rs=pst.executeQuery();
            list=qp.parseBatch(rs);
        }
        catch(Exception e) {throw new RuntimeException(e);}
        finally
        {
            try
            {
                if(rs!=null) rs.close();
                if(pst!=null) pst.close();
                if(con!=null) nc.releaseConnection(con);
            }
            catch(Exception e) {throw new RuntimeException(e);}
        }
        return list;
    }
    @Passed
    public <T> LinkedList<T> select(Class clazz, String[] colName)
    {
        LinkedList<T> list=null;
        Connection con=null;
        PreparedStatement pst=null;
        ResultSet rs=null;
        try
        {
            QueryParser qp=map.getParserOrLoad(clazz);
            con=nc.newConnection();
            pst=con.prepareStatement(Query.createSelectQuery(qp, colName));
            rs=pst.executeQuery();
            list=qp.parseBatch(rs, colName);
        }
        catch(Exception e) {throw new RuntimeException(e);}
        finally
        {
            try
            {
                if(rs!=null) rs.close();
                if(pst!=null) pst.close();
                if(con!=null) nc.releaseConnection(con);
            }
            catch(Exception e) {throw new RuntimeException(e);}
        }
        return list;
    }
    @Passed
    public <T> LinkedList<T> select(Class clazz, String condition)
    {
        LinkedList<T> list=null;
        Connection con=null;
        PreparedStatement pst=null;
        ResultSet rs=null;
        try
        {
            QueryParser qp=map.getParserOrLoad(clazz);
            con=nc.newConnection();
            pst=con.prepareStatement(Query.createSelectQuery(qp,condition));
            rs=pst.executeQuery();
            list=qp.parseBatch(rs);
        }
        catch(Exception e) {throw new RuntimeException(e);}
        finally
        {
            try
            {
                if(rs!=null) rs.close();
                if(pst!=null) pst.close();
                if(con!=null) nc.releaseConnection(con);
            }
            catch(Exception e) {throw new RuntimeException(e);}
        }
        return list;
    }
    @Passed
    public <T> LinkedList<T> select(Class clazz, String[] colName, String condition)
    {
        LinkedList<T> list=null;
        Connection con=null;
        PreparedStatement pst=null;
        ResultSet rs=null;
        try
        {
            QueryParser qp=map.getParserOrLoad(clazz);
            con=nc.newConnection();
            pst=con.prepareStatement(Query.createSelectQuery(qp,colName,condition));
            rs=pst.executeQuery();
            list=qp.parseBatch(rs, colName);
        }
        catch(Exception e) {throw new RuntimeException(e);}
        finally
        {
            try
            {
                if(rs!=null) rs.close();
                if(pst!=null) pst.close();
                if(con!=null) nc.releaseConnection(con);
            }
            catch(Exception e) {throw new RuntimeException(e);}
        }
        return list;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Ecapsulated:insert">
    @Passed
    public int insert(Class clazz, Object val)
    {
        int r=0;
        Connection con=null;
        PreparedStatement pst=null;
        try
        {
            QueryParser qp=map.getParserOrLoad(clazz);
            con=nc.newConnection();
            pst=con.prepareStatement(Query.createInsertQuery(qp));
            qp.setInsert(pst, val);
            r=pst.executeUpdate();
        }
        catch(Exception e) {throw new RuntimeException(e);}
        finally
        {
            try
            {
                if(pst!=null) pst.close();
                if(con!=null) nc.releaseConnection(con);
            }
            catch(Exception e) {throw new RuntimeException(e);}
        }
        return r;
    }
    @Passed
    public int insert(Class clazz, Object val, String[] colName)
    {
        int r=0;
        Connection con=null;
        PreparedStatement pst=null;
        try
        {
            con=nc.newConnection();
            QueryParser qp=map.getParserOrLoad(clazz);
            pst=con.prepareStatement(Query.createInsertQuery(qp, colName));
            qp.setInsert(pst, val, colName);
            r=pst.executeUpdate();
        }
        catch(Exception e) {throw new RuntimeException(e);}
        finally
        {
            try
            {
                if(pst!=null) pst.close();
                if(con!=null) nc.releaseConnection(con);
            }
            catch(Exception e) {throw new RuntimeException(e);}
        }
        return r;
    }
    @Passed
    public int insert(Class clazz, Collection batch)
    {
        int r=0;
        Connection con=null;
        PreparedStatement pst=null;
        try
        {
            QueryParser qp=map.getParserOrLoad(clazz);
            con=nc.newConnection();
            pst=con.prepareStatement(Query.createInsertQuery(qp));
            qp.setInsertBatch(pst, batch);
            r=Vector.sum(pst.executeBatch());
        }
        catch(Exception e) {throw new RuntimeException(e);}
        finally
        {
            try
            {
                if(pst!=null) pst.close();
                if(con!=null) nc.releaseConnection(con);
            }
            catch(Exception e) {e.printStackTrace();}
        }
        return r;
    }
    @Passed
    public int insert(Class clazz, Collection batch, String[] colName)
    {
        int r=0;
        Connection con=null;
        PreparedStatement pst=null;
        try
        {
            QueryParser qp=map.getParserOrLoad(clazz);
            con=nc.newConnection();
            pst=con.prepareStatement(Query.createInsertQuery(qp, colName));
            qp.setInsertBatch(pst, batch, colName);
             r=Vector.sum(pst.executeBatch());
        }
        catch(Exception e) {throw new RuntimeException(e);}
        finally
        {
            try
            {
                if(pst!=null) pst.close();
                if(con!=null) nc.releaseConnection(con);
            }
            catch(Exception e) {throw new RuntimeException(e);}
        }
        return r;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Ecapsulated:Update">
    @Passed
    public int update(Class clazz, Object val)
    {
        int r=0;
        Connection con=null;
        PreparedStatement pst=null;
        try
        {
            QueryParser qp=map.getParserOrLoad(clazz);
            con=nc.newConnection();
            pst=con.prepareStatement(Query.createUpdateQuery(qp));
            qp.setUpdate(pst, val);
            r=pst.executeUpdate();
        }
        catch(Exception e) {throw new RuntimeException(e);}
        finally
        {
            try
            {
                if(pst!=null) pst.close();
                if(con!=null) nc.releaseConnection(con);
            }
            catch(Exception e) {e.printStackTrace();}
        }
        return r;
    }
    @Passed
    public int update(Class clazz, Object val, String[] colName)
    {
         int r=0;
        Connection con=null;
        PreparedStatement pst=null;
        try
        {
            QueryParser qp=map.getParserOrLoad(clazz);
            con=nc.newConnection();
            pst=con.prepareStatement(Query.createUpdateQuery(qp,colName));
            qp.setUpdate(pst, val, colName);
            r=pst.executeUpdate();
        }
        catch(Exception e) {throw new RuntimeException(e);}
        finally
        {
            try
            {
                if(pst!=null) pst.close();
                if(con!=null) nc.releaseConnection(con);
            }
            catch(Exception e) {throw new RuntimeException(e);}
        }
        return r;
    }
    @Passed
    public int update(Class clazz, Collection batch)
    {
         int r=0;
        Connection con=null;
        PreparedStatement pst=null;
        try
        {
            QueryParser qp=map.getParserOrLoad(clazz);
            con=nc.newConnection();
            pst=con.prepareStatement(Query.createUpdateQuery(qp));
            qp.setUpdateBatch(pst, batch);
            r=pst.executeUpdate();
        }
        catch(Exception e) {throw new RuntimeException(e);}
        finally
        {
            try
            {
                if(pst!=null) pst.close();
                if(con!=null) nc.releaseConnection(con);
            }
            catch(Exception e) {e.printStackTrace();}
        }
        return r;
    }
    @Passed
    public int update(Class clazz, Collection batch, String[] colName)
    {
        int r=0;
        Connection con=null;
        PreparedStatement pst=null;
        try
        {
            QueryParser qp=map.getParserOrLoad(clazz);
            con=nc.newConnection();
            pst=con.prepareStatement(Query.createUpdateQuery(qp, colName));
            qp.setUpdateBatch(pst, batch, colName);
            r=pst.executeUpdate();
        }
        catch(Exception e) {throw new RuntimeException(e);}
        finally
        {
            try
            {
                if(pst!=null) pst.close();
                if(con!=null) nc.releaseConnection(con);
            }
            catch(Exception e) {throw new RuntimeException(e);}
        }
        return r;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Ecapsulated:Delete">
    @Passed
    public int delete(Class clazz, Object val)
    {
        int r=0;
        Connection con=null;
        PreparedStatement pst=null;
        try
        {
            QueryParser qp=map.getParserOrLoad(clazz);
            con=nc.newConnection();
            pst=con.prepareStatement(Query.createDeleteQuery(qp));
            qp.setDelete(pst, val);
            r=pst.executeUpdate();
        }
        catch(Exception e) {throw new RuntimeException(e);}
        finally
        {
            try
            {
                if(pst!=null) pst.close();
                if(con!=null) nc.releaseConnection(con);
            }
            catch(Exception e) {throw new RuntimeException(e);}
        }
        return r;
    }
    @Passed
    public int delete(Class clazz, Collection batch)
    {
        int r=0;
        Connection con=null;
        PreparedStatement pst=null;
        try
        {
            QueryParser qp=map.getParserOrLoad(clazz);
            con=nc.newConnection();
            pst=con.prepareStatement(Query.createDeleteQuery(qp));
            qp.setDeleteBatch(pst, batch);
            r=pst.executeUpdate();
        }
        catch(Exception e) {throw new RuntimeException(e);}
        finally
        {
            try
            {
                if(pst!=null) pst.close();
                if(con!=null) nc.releaseConnection(con);
            }
            catch(Exception e) {throw new RuntimeException(e);}
        }
        return r;
    }
    //</editor-fold>
    //</editor-fold>
}
