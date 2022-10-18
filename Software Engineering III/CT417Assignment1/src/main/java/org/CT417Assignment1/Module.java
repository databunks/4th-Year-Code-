package org.CT417Assignment1;

import java.util.ArrayList;

public class Module
{
    private String name;
    private String id;
    private ArrayList<Student> students;
    private ArrayList<Course> courses;
    private Lecturer lecturer;

    public Module(String name, String id, ArrayList<Student> students, ArrayList<Course> courses, Lecturer lecturer)
    {
        this.name = name;
        this.id = id;
        this.students = students;
        this.courses = courses;
        this.lecturer = lecturer;
    }

    public String GetName()
    {
        return this.name;
    }

    public String GetId()
    {
        return this.id;
    }

    public ArrayList<Student> GetStudents()
    {
        return this.students;
    }

    public ArrayList<Course> GetCourses()
    {
        return this.courses;
    }

    public Lecturer GetLecturer()
    {
        return this.lecturer;
    }




}