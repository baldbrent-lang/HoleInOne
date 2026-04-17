export function Brand({ subtitle }) {
  return (
    <div className="brand">
      <div className="logo" aria-hidden="true">
        <FlagIcon />
      </div>
      <div>
        <h1>Par One</h1>
        {subtitle && <div className="tag">{subtitle}</div>}
      </div>
    </div>
  );
}

export function FlagIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M7 4v17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M7 4l11 3.2L7 10.4" fill="currentColor" />
      <circle cx="7" cy="21" r="1.8" fill="currentColor" />
    </svg>
  );
}

export function Icon({ name, size = 20 }) {
  const paths = {
    camera: (
      <>
        <path d="M4 7a2 2 0 0 1 2-2h2l1.5-2h5L16 5h2a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7Z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
        <circle cx="12" cy="13" r="3.5" stroke="currentColor" strokeWidth="1.6" />
      </>
    ),
    check: <path d="M5 12.5l4.2 4.2L19 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />,
    download: (
      <>
        <path d="M12 4v12m0 0l-4-4m4 4l4-4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M5 20h14" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      </>
    ),
    share: (
      <>
        <circle cx="6" cy="12" r="2.6" stroke="currentColor" strokeWidth="1.6" />
        <circle cx="18" cy="6" r="2.6" stroke="currentColor" strokeWidth="1.6" />
        <circle cx="18" cy="18" r="2.6" stroke="currentColor" strokeWidth="1.6" />
        <path d="M8.3 10.8l7.4-3.6M8.3 13.2l7.4 3.6" stroke="currentColor" strokeWidth="1.6" />
      </>
    ),
    flag: (
      <>
        <path d="M6 3v18" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        <path d="M6 4h11l-2 3 2 3H6" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
      </>
    ),
    qr: (
      <>
        <path d="M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4z" stroke="currentColor" strokeWidth="1.6" />
        <path d="M14 14h2v2h-2zM18 14h2v2h-2zM14 18h2v2h-2zM18 18h2v2h-2z" fill="currentColor" />
      </>
    ),
    chart: (
      <>
        <path d="M4 20V8m6 12V4m6 16v-8m6 8V12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      </>
    ),
    dollar: (
      <>
        <path d="M12 3v18" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        <path d="M16 7.5C16 6.1 14 5 12 5s-4 1-4 3 2 2.5 4 3 4 1 4 3-2 3-4 3-4-1.1-4-2.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      </>
    ),
    users: (
      <>
        <circle cx="9" cy="9" r="3" stroke="currentColor" strokeWidth="1.6" />
        <path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        <circle cx="17" cy="8" r="2.5" stroke="currentColor" strokeWidth="1.6" />
        <path d="M21 19c0-2.6-2-4.5-4.5-4.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </>
    ),
    sparkle: <path d="M12 3l1.8 5.4L19 10l-5.2 1.6L12 17l-1.8-5.4L5 10l5.2-1.6L12 3z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />,
    clock: (
      <>
        <circle cx="12" cy="12" r="8" stroke="currentColor" strokeWidth="1.6" />
        <path d="M12 8v4l2.5 2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </>
    ),
    lock: (
      <>
        <rect x="5" y="11" width="14" height="9" rx="2" stroke="currentColor" strokeWidth="1.6" />
        <path d="M8 11V8a4 4 0 0 1 8 0v3" stroke="currentColor" strokeWidth="1.6" />
      </>
    ),
    retake: (
      <>
        <path d="M4 12a8 8 0 0 1 14-5.3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        <path d="M18 3v4h-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M20 12a8 8 0 0 1-14 5.3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        <path d="M6 21v-4h4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </>
    ),
    flip: (
      <>
        <path d="M12 4v16" stroke="currentColor" strokeWidth="1.6" strokeDasharray="2 3" />
        <path d="M4 8l-2 2 2 2M4 10h6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M20 16l2-2-2-2M20 14h-6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </>
    ),
    capture: (
      <>
        <circle cx="12" cy="12" r="8" stroke="currentColor" strokeWidth="1.6" />
        <circle cx="12" cy="12" r="4" fill="currentColor" />
      </>
    ),
  };
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      {paths[name]}
    </svg>
  );
}
