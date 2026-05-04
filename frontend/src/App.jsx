import { Route, Routes, Link, useLocation } from "react-router-dom";
import Register from "./pages/Register.jsx";
import Confirmation from "./pages/Confirmation.jsx";
import Gallery from "./pages/Gallery.jsx";
import Admin from "./pages/Admin.jsx";
import AdminParticipants from "./pages/AdminParticipants.jsx";
import AdminReview from "./pages/AdminReview.jsx";
import AdminUpload from "./pages/AdminUpload.jsx";
import AdminShowcase from "./pages/AdminShowcase.jsx";
import Login from "./pages/Login.jsx";
import Signup from "./pages/Signup.jsx";
import Me from "./pages/Me.jsx";
import CoursePicker from "./pages/CoursePicker.jsx";
import SampleGallery from "./pages/SampleGallery.jsx";
import Legal from "./pages/Legal.jsx";
import Invite from "./pages/Invite.jsx";
import Pay from "./pages/Pay.jsx";
import Leaderboards from "./pages/Leaderboards.jsx";
import Footer from "./components/Footer.jsx";
import Home from "./pages/Home.jsx";

function ConditionalFooter() {
  const { pathname } = useLocation();
  // No footer on the in-page registration / camera flows or admin pages.
  if (pathname.startsWith("/admin")) return null;
  if (pathname.startsWith("/r/")) return null;
  return (
    <div className="wrap" style={{ paddingTop: 0, paddingBottom: 32 }}>
      <Footer />
    </div>
  );
}

export default function App() {
  return (
    <>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/me" element={<Me />} />
        <Route path="/courses" element={<CoursePicker />} />
        <Route path="/sample" element={<SampleGallery />} />
        <Route path="/legal/:doc" element={<Legal />} />
        <Route path="/invite/:galleryToken" element={<Invite />} />
        <Route path="/pay/:participantId" element={<Pay />} />
        <Route path="/leaderboards" element={<Leaderboards />} />
        <Route path="/r/:courseToken" element={<Register />} />
        <Route path="/confirm/:participantId" element={<Confirmation />} />
        <Route path="/g/:galleryToken" element={<Gallery />} />
        <Route path="/admin" element={<Admin />} />
        <Route path="/admin/participants" element={<AdminParticipants />} />
        <Route path="/admin/upload" element={<AdminUpload />} />
        <Route path="/admin/showcase" element={<AdminShowcase />} />
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
      <ConditionalFooter />
    </>
  );
}
