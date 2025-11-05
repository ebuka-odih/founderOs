import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import Workspace from "../app/page";

const flushPromises = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("FounderOs workspace", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    global.fetch = vi.fn() as unknown as typeof fetch;
  });

  it("navigates between stages from the sidebar", () => {
    render(<Workspace />);

    const problemStageButton = screen.getByRole("button", { name: /Problem & Solution Fit/i });
    fireEvent.click(problemStageButton);

    expect(screen.getByText(/Stage: Problem Fit Assessment/i)).toBeInTheDocument();
  });

  it("retains shared input across stages", () => {
    render(<Workspace />);

    fireEvent.change(screen.getByPlaceholderText(/Describe your idea/i), {
      target: { value: "Shared context for all stages." }
    });

    const problemStageButton = screen.getByRole("button", { name: /Problem & Solution Fit/i });
    fireEvent.click(problemStageButton);

    expect(screen.getByPlaceholderText(/What problem do you solve/i)).toHaveValue(
      "Shared context for all stages."
    );
  });

  it("submits input to the backend and displays returned HTML", async () => {
    const mockHtml = '<section class="stage-report" data-stage="idea">Mock HTML</section>';
    (global.fetch as unknown as vi.Mock).mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(mockHtml)
    });

    render(<Workspace />);

    fireEvent.change(screen.getByPlaceholderText(/Describe your idea/i), {
      target: { value: "We connect artisans with clients through trusted profiles." }
    });

    fireEvent.change(screen.getByPlaceholderText(/e\.g\. market sizing/i), {
      target: { value: "trust signals, onboarding" }
    });

    fireEvent.click(screen.getByText(/Generate Report/i));

    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));

    const [url, options] = (global.fetch as vi.Mock).mock.calls[0];
    expect(url).toBe("http://localhost:8000/validate/idea");
    expect(options?.method).toBe("POST");

    const payload = JSON.parse(options?.body ?? "{}");
    expect(payload.idea_overview).toContain("connect artisans");
    expect(payload.focus_points).toEqual(["trust signals", "onboarding"]);

    await flushPromises();

    expect(screen.getByTestId("stage-html")).toContainHTML(mockHtml);
  });
});
