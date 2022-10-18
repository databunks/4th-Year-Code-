import org.CT417Assignment1.Course;
import org.CT417Assignment1.Lecturer;
import org.CT417Assignment1.Module;
import org.CT417Assignment1.Student;
import org.joda.time.*;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.sql.Array;
import java.util.ArrayList;

public class TestUniversityClasses {

    @Test
    void TestStudent()
    {
        String studentName = "Jim Bob";
        int studentAge = 20;

        DateTime studentDob = new DateTime(2002, 3, 12, 0, 0, 0);

        Course testCourse = new Course(null, null, null, null, null);
        ArrayList<Course> testCourses = new ArrayList<Course>();
        testCourses.add(testCourse);

        Module testModule = new Module(null, null, null, null, null);
        ArrayList<Module> testModules = new ArrayList<Module>();
        testModules.add(testModule);

        Student testStudent = new Student(studentName, studentAge, studentDob, testCourses, testModules);

        // Verifying the age of the student object is valid
        Years expectedAge = Years.yearsBetween(studentDob, DateTime.now());
        Assertions.assertEquals(expectedAge.getYears(), testStudent.GetAge());

        Assertions.assertEquals(testStudent.GetName(), studentName);
        Assertions.assertEquals(testStudent.GetAge(), studentAge);
        Assertions.assertEquals(testStudent.GetDateOfBirth(), studentDob);

        // Whenever we create a student the static ID is updated which starts at 0 with 2 students being created the id should be 2 and not 1 so we check for that
        Student testStudent2 = new Student(studentName, studentAge, studentDob, testCourses, testModules);
        Assertions.assertNotEquals(1, testStudent.GetId());

        // Verifying the username is in the correct format
        String nameWithoutSpaces = studentName.replaceAll("\\s+", "");
        String expectedUsername = nameWithoutSpaces + studentAge;
        Assertions.assertEquals(expectedUsername, testStudent.GetUsername());

        Assertions.assertEquals(testCourses, testStudent.GetCourses());
        Assertions.assertEquals(testModules, testStudent.GetModules());

    }

    @Test
    void TestLecturer()
    {
        String lecturerName = "Bob marley";
        int lecturerAge = 32;

        DateTime lecturerDob = new DateTime(1990, 3, 12, 0, 0, 0);

        Module testModule = new Module(null, null, null, null, null);
        ArrayList<Module> testModules = new ArrayList<Module>();
        testModules.add(testModule);

        Lecturer testLecturer = new Lecturer(lecturerName, lecturerAge, lecturerDob, testModules);

        // Verifying the age of the student object is valid
        Years expectedAge = Years.yearsBetween(lecturerDob, DateTime.now());
        Assertions.assertEquals(expectedAge.getYears(), testLecturer.GetAge());

        Assertions.assertEquals(testLecturer.GetName(), lecturerName);
        Assertions.assertEquals(testLecturer.GetAge(), lecturerAge);
        Assertions.assertEquals(testLecturer.GetDateOfBirth(), lecturerDob);

        // Whenever we create a Lecturer the static ID is updated which starts at 0 with 2 students being created the id should be 2 and not 1 so we check for that
        Lecturer testLecturer2 = new Lecturer(lecturerName, lecturerAge, lecturerDob, testModules);
        Assertions.assertNotEquals(1, testLecturer.GetId());

        // Verifying the username is in the correct format
        String nameWithoutSpaces = lecturerName.replaceAll("\\s+", "");
        String expectedUsername = nameWithoutSpaces + lecturerAge;
        Assertions.assertEquals(expectedUsername, testLecturer.GetUsername());

        Assertions.assertEquals(testModules, testLecturer.GetModules());
    }

    @Test
    void TestModule() {
        String moduleName = "Software Engineering";
        String moduleId = "CT-435";

        Student testStudent = new Student("wa", 32, null, null, null);
        ArrayList<Student> testStudents = new ArrayList<Student>();
        testStudents.add(testStudent);

        Course testCourse = new Course(null, null, null, null, null);
        ArrayList<Course> testCourses = new ArrayList<Course>();
        testCourses.add(testCourse);

        Lecturer testLecturer = new Lecturer("e", 0, null, null);

        Module testModule = new Module(moduleName, moduleId, testStudents, testCourses, testLecturer);

        Assertions.assertEquals(moduleName, testModule.GetName());
        Assertions.assertEquals(moduleId, testModule.GetId());
        Assertions.assertEquals(testStudents, testModule.GetStudents());
        Assertions.assertEquals(testCourses, testModule.GetCourses());
        Assertions.assertEquals(testLecturer, testModule.GetLecturer());

    }

    @Test
    void TestCourse()
    {
        String courseName = "Computer Science";

        Module testModule = new Module(null,null,null,null,null);
        ArrayList<Module> testModules = new ArrayList<Module>();
        testModules.add(testModule);

        Module testStudent = new Module(null,null,null,null,null);
        ArrayList<Student> testStudents = new ArrayList<Student>();
        testModules.add(testStudent);

        DateTime courseStartDate = new DateTime(2021,5,4,3,6);
        DateTime courseEndDate = new DateTime(2025,5,4,3,6);

        Course testCourse = new Course(courseName,testModules,testStudents,courseStartDate,courseEndDate);

        Assertions.assertEquals(courseName, testCourse.GetName());
        Assertions.assertEquals(testModules, testCourse.GetModules());
        Assertions.assertEquals(courseStartDate, testCourse.GetStartDate());
        Assertions.assertEquals(courseEndDate, testCourse.GetEndDate());

    }

}


