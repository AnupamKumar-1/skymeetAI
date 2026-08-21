import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { isValidToken } from "./authToken";

const withAuth = (WrappedComponent) => {
  const AuthComponent = (props) => {
    const router = useNavigate();
    const [isAllowed, setIsAllowed] = useState(null);
    const isAuthenticated = () => isValidToken(localStorage.getItem("token"));

    useEffect(() => {
      if (!isAuthenticated()) {
        console.warn("User not authenticated → redirecting");

        setIsAllowed(false);

        router("/auth", { replace: true });
      } else {
        setIsAllowed(true);
      }
    }, [router]);


    if (isAllowed === null) return null;


    if (!isAllowed) return null;


    return <WrappedComponent {...props} />;
  };

  return AuthComponent;
};

export default withAuth;