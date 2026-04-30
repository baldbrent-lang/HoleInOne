import { Route, Routes, Link } from "react-router-dom";
import Register from "./pages/Register.jsx";
import Confirmation from "./pages/Confirmation.jsx";
import Gallery from "./pages/Gallery.jsx";
import Admin from "./pages/Admin.jsx";
import AdminParticipants from "./pages/AdminParticipants.jsx";
import AdminReview from "./pages/AdminReview.jsx";
import AdminUpload from "./pages/AdminUpload.jsx";
import Login from "./pages/Login.jsx";
import Signup from "./pages/Signup.jsx";
import Me from "./pages/Me.jsx";
import CoursePicker from "./pages/CoursePicker.jsx";
import Home from "./pages/Home.jsx";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/me" element={<Me />} />
      <Route path="/courses" element={<CoursePicker />} />
      <Route path="/r/:courseToken" element={<Register />} />
      <Route path="/confirm/:participantId" element={<Confirmation />} />
      <Route path="/g/:galleryToken" element={<Gallery />} />
      <Route path="/admin" element={<Admin />} />
      <Route path="/admin/participants" element={<AdminParticipants />} />
      <Route path="/admin/upload" element={<AdminUpload />} />
      <Route path="/admin/review" element={<AdminReview />} />
      <Route
        path="*"
        element={
          <div className="wrap">
            <div className="card">
              <h1>Not found</h1>
              <Link to="/">Go home</Link>
            </div>
          </div>
        }
      />
    </Routes>
  );
}
